cmake_minimum_required(VERSION 3.16)

project(BioFilmZ VERSION 0.1 LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python3 and its components
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

if(Python3_FOUND)
  message(STATUS "Python found at: ${Python3_EXECUTABLE}")
else()
  message(FATAL_ERROR "Python not found!")
endif()

# Add include directories for Python
include_directories(${Python3_INCLUDE_DIRS})

# Set up Qt
find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets)

set(PROJECT_SOURCES
    main.cpp
    mainwindow.cpp
    mainwindow.h
    mainwindow.ui
)

if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
    qt_add_executable(BioFilmZ
        MANUAL_FINALIZATION
        ${PROJECT_SOURCES}
    )
else()
    if(ANDROID)
        add_library(BioFilmZ SHARED
            ${PROJECT_SOURCES}
        )
    else()
        add_executable(BioFilmZ
            ${PROJECT_SOURCES}
        )
    endif()
endif()

# Link libraries for Qt and Python
target_link_libraries(BioFilmZ PRIVATE Qt${QT_VERSION_MAJOR}::Widgets ${Python3_LIBRARIES})

# Set target properties
set_target_properties(BioFilmZ PROPERTIES
    MACOSX_BUNDLE TRUE
    WIN32_EXECUTABLE TRUE
)

include(GNUInstallDirs)
install(TARGETS BioFilmZ
    BUNDLE DESTINATION .
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(QT_VERSION_MAJOR EQUAL 6)
    qt_finalize_executable(BioFilmZ)
endif()
