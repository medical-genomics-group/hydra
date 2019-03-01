cmake_minimum_required(VERSION 2.8.12)
add_subdirectory(src)

if(TEST)
	add_subdirectory(test)
endif()
	