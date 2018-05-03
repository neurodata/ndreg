cmake_minimum_required(VERSION 3.5)
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

#set(CMAKE_RUNTIME_OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

include(${ITK_USE_FILE})

add_executable(metamorphosis metamorphosis.cxx itkMetamorphosisImageRegistrationMethodv4.h itkMetamorphosisImageRegistrationMethodv4.hxx)
target_link_libraries(metamorphosis ${ITK_LIBRARIES})
