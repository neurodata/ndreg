cmake_minimum_required(VERSION 3.1)
project(ndreg)

set(PACKAGE_VERSION "0.1")

# Build metamorphosis binary
find_package(ITK)
if(NOT ITK_FOUND)
    message(FATAL_ERROR "ITK not found. Please set ITK_DIR.")
endif()

find_program(PYTHON "python")
if(NOT PYTHON)
    message(FATAL_ERROR "Python not found. Please set PYTHON.")
endif()

# Configure setup.py in binary directory
#set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
#set(SETUP_PY "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
#configure_file(${SETUP_PY_IN} ${SETUP_PY})

# Copy python source files to binary directory
#set(PY_FILENAME_LIST "ndreg.py;__init__.py")
#
#foreach(FILENAME ${PY_FILENAME_LIST})
#  set(SRC_PY_PATH ${CMAKE_CURRENT_SOURCE_DIR}/ndreg/${FILENAME})
#  set(BIN_PY_PATH ${CMAKE_CURRENT_BINARY_DIR}/ndreg/${FILENAME})
#  set(BIN_PY_PATH_LIST ${BIN_PY_PATH_LIST} ${BIN_PY_PATH})
#
#  add_custom_command(OUTPUT ${BIN_PY_PATH}
#    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC_PY_PATH} ${BIN_PY_PATH}
#    DEPENDS ${SRC_PY_PATH})
#endforeach()

# Build ndreg egg
include(${ITK_USE_FILE})
#add_subdirectory(ndreg)

# stuff for ndreg access
#add_executable(metamorphosis metamorphosis.cxx itkMetamorphosisImageRegistrationMethodv4.h itkMetamorphosisImageRegistrationMethodv4.hxx)
#target_link_libraries(metamorphosis ${ITK_LIBRARIES})

#set(TIMESTAMP "${CMAKE_CURRENT_BINARY_DIR}/build/timestamp")
#add_custom_command(OUTPUT ${TIMESTAMP}
#    COMMAND ${PYTHON} -m py_compile ${BIN_PY_PATH_LIST} # Check for syntax errors by compiling python files
#    COMMAND ${PYTHON} ${SETUP_PY} build                 # Make the egg
#    COMMAND ${CMAKE_COMMAND} -E touch ${TIMESTAMP}      # Generate timestamp file
#    DEPENDS ${BIN_PY_PATH_LIST} metamorphosis)
#
#add_custom_target(ndreg ALL DEPENDS ${TIMESTAMP})
#
# Install ndreg egg
#install(CODE "execute_process(COMMAND ${PYTHON} ${SETUP_PY} install --user)")
add_executable(metamorphosis metamorphosis.cxx itkMetamorphosisImageRegistrationMethodv4.h itkMetamorphosisImageRegistrationMethodv4.hxx)
target_link_libraries(metamorphosis ${ITK_LIBRARIES})
