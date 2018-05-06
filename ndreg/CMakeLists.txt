cmake_minimum_required(VERSION 3.5)
project(ndreg)

set(PACKAGE_VERSION "0.1")

# Build metamorphosis binary
find_package(ITK)
if(NOT ITK_FOUND)
	#-----------------------------------------------------------------------------
	include(ExternalProject)

	set(ITK_REPOSITORY "https://github.com/InsightSoftwareConsortium/ITK.git")
	# ITK nightly-master 2018-05-02
	set(ITK_GIT_TAG "dbeee5a")

	#-----------------------------------------------------------------------------
	# A separate project is used to download ITK, so that it can reused
	# when building different "flavor" of ITK python packages

	message(STATUS "SuperBuild -")
	message(STATUS "SuperBuild - ITK-source-download")

	# Sanity checks
	if(DEFINED ITK_SOURCE_DIR AND NOT EXISTS ${ITK_SOURCE_DIR})
	message(FATAL_ERROR "ITK_SOURCE_DIR variable is defined but corresponds to nonexistent directory")
	endif()

	if(NOT DEFINED ITK_SOURCE_DIR)

	set(ITK_SOURCE_DIR ${CMAKE_BINARY_DIR}/ITK-source)

	ExternalProject_add(ITK-source-download
	  SOURCE_DIR ${ITK_SOURCE_DIR}
	  GIT_REPOSITORY ${ITK_REPOSITORY}
	  GIT_TAG ${ITK_GIT_TAG}
	  USES_TERMINAL_DOWNLOAD 1
	  CONFIGURE_COMMAND ""
	  BUILD_COMMAND ""
	  INSTALL_COMMAND ""
	  )
	set(proj_status "")

	else()

	ipp_ExternalProject_Add_Empty(
	  ITK-source-download
	  ""
	  )
	set(proj_status " (REUSE)")

	endif()

	message(STATUS "SuperBuild -   ITK_SOURCE_DIR: ${ITK_SOURCE_DIR}")
	message(STATUS "SuperBuild - ITK-source-download[OK]${proj_status}")

    #message(FATAL_ERROR "ITK not found. Please set ITK_DIR.")
endif()

find_program(PYTHON "python")
if(NOT PYTHON)
    message(FATAL_ERROR "Python not found. Please set PYTHON.")
endif()

#set(CMAKE_RUNTIME_OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

include(${ITK_USE_FILE})

add_executable(metamorphosis metamorphosis.cxx itkMetamorphosisImageRegistrationMethodv4.h itkMetamorphosisImageRegistrationMethodv4.hxx)
target_link_libraries(metamorphosis ${ITK_LIBRARIES})
