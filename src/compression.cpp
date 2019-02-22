#include "compression.h"

#include <zlib.h>
#include <iostream>

unsigned long maxCompressedDataSize(const unsigned int numFloats)
{
    // Initialise zlib
    z_stream strm;
    strm.zalloc = nullptr;
    strm.zfree = nullptr;
    strm.opaque = nullptr;
    const int level = -1;
    auto ret = deflateInit(&strm, level);
    if (ret != Z_OK)
        return 0;

    // Calculate the maximum buffer size needed to hold the compressed data
    const unsigned int inputSize = numFloats * sizeof(float);
    strm.avail_in = inputSize;
    const auto maxOutputSize = deflateBound(&strm, inputSize);
    //std::cout << "maxSize = " << maxOutputSize << " bytes = " << maxOutputSize / 1024 << " KiB" << std::endl;

    // Clean up
    (void) deflateEnd(&strm);

    return maxOutputSize;
}

unsigned long compressData(const VectorXd &snpData, unsigned char *outputBuffer, unsigned long outputSize)
{
    // Initialise zlib
    z_stream strm;
    strm.zalloc = nullptr;
    strm.zfree = nullptr;
    strm.opaque = nullptr;
    const int level = -1;
    auto ret = deflateInit(&strm, level);
    if (ret != Z_OK)
        return 0;

    // Compress the data
    const unsigned int inputSize = static_cast<unsigned int>(snpData.size()) * sizeof(float);
    strm.avail_in = inputSize;
    strm.next_in = reinterpret_cast<unsigned char *>(const_cast<double*>(&snpData[0]));
    strm.avail_out = static_cast<unsigned int>(outputSize);
    strm.next_out = outputBuffer;

    const int flush = Z_FINISH;
    ret = deflate(&strm, flush);
    if (ret != Z_STREAM_END) {
        std::cout << "Error compressing data" << std::endl;
        return 0;
    }
    const auto compressedSize = outputSize - strm.avail_out;
    // std::cout << "compressedSize = " << compressedSize << " bytes = "
    //           << compressedSize / 1024 << " KiB" << std::endl;

    // Clean up
    (void) deflateEnd(&strm);

    // DEBUG: Verify compressed data can be decompressed to reproduce the original data
    /*
    z_stream strm2;
    strm2.zalloc = nullptr;
    strm2.zfree = nullptr;
    strm2.opaque = nullptr;
    strm2.avail_in = 0;
    strm2.next_in = nullptr;
    ret = inflateInit(&strm2);
    if (ret != Z_OK) {
        std::cout << "Failed to verify compressed data" << std::endl;
        return compressedSize;
    }
    unsigned char *checkBuffer = new unsigned char[inputSize];
    strm2.next_out = checkBuffer;
    strm2.avail_out = inputSize;
    strm2.next_in = outputBuffer;
    strm2.avail_in = static_cast<unsigned int>(compressedSize);
    ret = inflate(&strm2, flush);
    if (ret != Z_STREAM_END) {
        std::cout << "Failed to verify compressed data" << std::endl;
        return compressedSize;
    }
    // Compare input and re-extracted data
    {
        Map<VectorXf> decompressedSnpData(reinterpret_cast<float *>(checkBuffer), snpData.size());
        for (int i = 0; i < snpData.size(); ++i) {
            const auto delta = snpData[i] - decompressedSnpData[i];
            std::cout << i << ": delta = " << delta << std::endl;
        }
    }
    // Cleanup
    delete[] checkBuffer;
    (void) inflateEnd(&strm2);
    */

    return compressedSize;
}

void extractData(unsigned char *compressedData,
                 unsigned int compressedDataSize,
                 unsigned char *outputBuffer,
                 unsigned int outputBufferSize)
{
    z_stream strm;
    strm.zalloc = nullptr;
    strm.zfree = nullptr;
    strm.opaque = nullptr;
    strm.avail_in = 0;
    strm.next_in = nullptr;
    auto ret = inflateInit(&strm);
    if (ret != Z_OK)
        throw("Failed to verify compressed data");

    strm.next_out = outputBuffer;
    strm.avail_out = outputBufferSize;
    strm.next_in = compressedData;
    strm.avail_in = compressedDataSize;
    const int flush = Z_FINISH;
    ret = inflate(&strm, flush);
    if (ret != Z_STREAM_END)
        throw("Failed to verify compressed data");

    (void) inflateEnd(&strm);
}
