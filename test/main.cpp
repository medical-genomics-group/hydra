/*
 * main.cpp
 *
 *  Created on: Jan 9, 2019
 *      Author: daniel
 */

#include <gtest/gtest.h>

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}

