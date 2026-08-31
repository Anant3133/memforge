cmake_minimum_required(VERSION 3.18)

add_executable(test_block_allocator test_block_allocator.cpp)
target_link_libraries(test_block_allocator PRIVATE memforge_core)

add_executable(test_scheduler test_scheduler.cpp)
target_link_libraries(test_scheduler PRIVATE memforge_core)

add_test(NAME test_block_allocator COMMAND test_block_allocator)
add_test(NAME test_scheduler COMMAND test_scheduler)
