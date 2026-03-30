add_executable(cuda_mad_tests test_main.cpp)

target_include_directories(cuda_mad_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

target_link_libraries(cuda_mad_tests PRIVATE ${OpenCV_LIBS})

add_test(NAME unit_tests COMMAND cuda_mad_tests)
