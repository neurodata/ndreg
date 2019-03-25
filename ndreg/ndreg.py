#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import SimpleITK as sitk
import tempfile
import shutil
from itertools import product
import tensorflow as tf
from . import util,  preprocessor, plotter, lddmm
import ndreg
from skimage.transform import downscale_local_mean
from tqdm import trange

def register_brain(I,xI,J,xJ,vt0,a=5.0,eV=1e-2,niter=50, naffine=0, outdir=None):
    """Register 3D mouse brain to the Allen Reference atlas using affine and deformable registration.

    Parameters:
    ----------
    atlas : {SimpleITK.SimpleITK.Image}
        Allen reference atlas or other atlas to register data to.
    img : {SimpleITK.SimpleITK.Image}
        Input observed 3D mouse brain volume
    outdir : {str}, optional
        Path to output directory to store intermediates. (the default is None, which will store all outputs in './')

    Returns
    -------
    SimpleITK.SimpleITK.Image
        The atlas deformed to fit the input image.
    """
#     params = dict()
#     params['x0I'] = np.arange(I.shape[0], dtype=float)
#     params['x1I'] = np.arange(I.shape[1], dtype=float)
#     params['x2I'] = np.arange(I.shape[2], dtype=float)
#     params['x0J'] = np.arange(J.shape[0], dtype=float)
#     params['x1J'] = np.arange(J.shape[1], dtype=float)
#     params['x2J'] = np.arange(J.shape[2], dtype=float)
#     params['a'] = 5.0
#     params['p'] = 2 # should be at least 2 in 3D
#     params['nt'] = 5
#     params['sigmaM'] = 1.0 # matching weight 1/2/sigma^2
#     params['sigmaA'] = 10.0 # matching weight for "artifact image"
#     params['sigmaR'] = 1.0 # regularization weight 1/2/sigma^2
#     params['eV'] = 1e-1 # step size for deformation parameters
#     params['eL'] = 0.0 # linear part of affine
#     params['eT'] = 0.0 # step size for translation part of affine
#     params['rigid'] = False # rigid only versus general affine
#     params['niter'] = 200 # iterations of gradient decent
#     params['naffine'] = 0 # do affine only for this number
#     params['post_affine_reduce'] = 0.1 # reduce affine step sizes by this much once nonrigid starts
#     params['nMstep'] = 0 # number of iterations of M step each E step in EM algorithm, 0 means don't use this feature
#     params['nMstep_affine'] = 0 # number of iterations of M step during affine
#     params['CA0'] = np.mean(J) # initial guess for value of artifact
#     params['W'] = 1.0 # a fixed weight for each pixel in J, or just a number
 
#     # initial guess
#     params['A0'] = np.eye(4)
#     # initial guess for velocity field
#     params['vt00'] = None
#     params['vt01'] = None
#     params['vt02'] = None
    if outdir is None: outdir = './'
    nxI = list(I.shape)
    nxJ = list(J.shape)
#     xI = [np.arange(nxi)*dxi - np.mean(np.arange(nxi)*dxi) for nxi,dxi in zip(nxI,dx)]
#     xJ = [np.arange(nxi)*dxi - np.mean(np.arange(nxi)*dxi) for nxi,dxi in zip(nxJ,dx)]

    # parameters
    # cost function weights 1 / sigma^2
    sigmaM = np.std(J) # matching
    sigmaA = sigmaM*10.0 # artifact
    sigmaR = 1e0 # regularization

    # enery operator, power of laplacian p, characteristic length a
    p = 2
#     a = (xI[0][1]-xI[0][0])*5

    # other optimization parameters
#     niter = 10 # how many iteraitons of gradient descent
#     naffine = 0 # first naffine iterations are affine only (no deformation)
    nt = 5 # this many timesteps to numerically integrate flow

    # When working with weights in EM algorithm, how many M steps per E step
    # first test with 0 (it is working)
    nMstep = 5
    nMstep_affine = 1

    # gradient descent step size
    eL = 2e-4
    eT = 5e-3
#     eV = 1e-2
    # there is some oscilation in the translation and the linear part
    if vt0 == None:
        vt0 = [None]*3
    out = lddmm.lddmm(I, J,
                  xI=xI, # location of pixels in domain
                  xJ=xJ,
                  niter=niter, # iterations of gradient descent
                  naffine=naffine, # iterations of affine only
                  eV = eV, # step size for deformation parameters
                  eT = eT, # step size for translation parameters
                  eL = eL, # step size for linear parameters
                  nt=nt, # timesteps for integtating flow
                  sigmaM=sigmaM, # matching cost weight 1/2sigmaM^2
                  sigmaR=sigmaR, # reg cost weight 1/2sigmaM^2
                  sigmaA=sigmaA, # artifact cost weight 1/2sigmaA^2
                  a=a, # kernel width
                  p=p, # power of laplacian in kernel (should be at least 2 for 3D)
#                  A0=A0, # initial guess for affine matrix (should get orientation right)
                  nMstep=nMstep, # number of m steps for each e step
                  nMstep_affine=nMstep_affine, # number of m steps during affine only phase
                  verbose=1,
                  vt00=vt0[0],
                  vt01=vt0[1],
                  vt02=vt0[2]
                 )
    util.write_lddmm_output(out,outdir)
    atlas_deformed = apply_transformation_to(I,xI,out['phiinvAinv0'],out['phiinvAinv1'],out['phiinvAinv2'])
    return atlas_deformed, out

def multi_res_lddmm(I,xI,J,xJ,resolutions=4):
    # make these in ascending order
    downsample_factors = [2**i for i in range(resolutions)][::-1]
    dI = [i[1]-i[0] for i in xI]
#    dJ = [i[1]-i[0] for i in xJ]
    spacings = [np.array(dI)*i for i in downsample_factors]
    # deformation step size
    eV = np.array(downsample_factors)*1e-2
    # kernel width
    # use the same one for all resolutions
    a = spacings[-1][0]*5.0
    shapes = []
    tmp_vt = None
    for i in trange(len(downsample_factors)):
        I_ds = downscale_local_mean(I,tuple([downsample_factors[i]]*3))
        # I would like to pad one slice of the allen atlas so that it has zero boundary conditions
        I_ds = np.pad(I_ds,((0,0),(0,0),(0,1)),'constant',constant_values=0)
        shapes.append(I_ds.shape)
        J_ds = downscale_local_mean(J,tuple([downsample_factors[i]]*3))
        nxI = list(I_ds.shape)
        nxJ = list(J_ds.shape)
        dx = spacings[i]
        xI = [np.arange(nxi)*dxi - np.mean(np.arange(nxi)*dxi) for nxi,dxi in zip(nxI,dx)]
        xJ = [np.arange(nxi)*dxi - np.mean(np.arange(nxi)*dxi) for nxi,dxi in zip(nxJ,dx)]
        # first resolution
        if i == 0:
            atlas_registered, out = register_brain(I_ds,xI, J_ds,xJ,tmp_vt,
                                                   eV=eV[i],a=a,niter=10,naffine=5)
            tmp_vt = [out['vt{}'.format(i)] for i in range(3)]
        else:
            tmp_vt2 = []
            old_pos = get_grid_locations(shapes[i-1],spacings[i-1])
            x2 = get_grid_locations(shapes[i],spacings[i])
            new_positions = np.meshgrid(x2[0],x2[1],x2[2],indexing='ij')
            new_shape = new_positions[0].shape
            for v in tmp_vt:
                tmp_t = np.zeros((new_shape[0],new_shape[1],new_shape[2],5),dtype='float32')
                for t in range(v.shape[-1]):
                    tmp_t[:,:,:,t] = resample(v[:,:,:,t],old_pos,new_positions)
    #             print("shape of upsampled velocity: {}".format(tmp_t.shape))
                tmp_vt2.append(tmp_t)

            tmp_vt = tmp_vt2
            atlas_registered, out = register_brain(I_ds,xI, J_ds,xJ,vt0=tmp_vt,
                                                   eV=eV[i],a=a,niter=50,naffine=0)
            tmp_vt = [out['vt{}'.format(i)] for i in range(3)]
    return atlas_registered, out

def apply_transformation_to(image,x,trans_x,trans_y,trans_z):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        Id = lddmm.interp3(x[0],x[1],x[2],image,trans_x,trans_y,trans_z)
        Idnp = Id.eval()
    return Idnp


def get_grid_locations(shape,spacing):
    x = [np.arange(nxi)*dxi - np.mean(np.arange(nxi)*dxi) for nxi,dxi in zip(list(shape),spacing)]
#     x_meshgrid = np.meshgrid(x[0],x[1],x[2],indexing='ij')
    return x
    
def resample(image,image_meshgrid,transformation):
#     grid_locations = get_meshgrid(image,spacing)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        image_r = lddmm.interp3(image_meshgrid[0],image_meshgrid[1],image_meshgrid[2],
                                image,
                                transformation[0],transformation[1],transformation[2])
        image_rnp = image_r.eval()
    return image_rnp