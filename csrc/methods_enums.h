#ifndef METHODS_ENUM_H
#define METHODS_ENUM_H
#include <stdio.h>
// Define the enum
typedef enum {
    ORIGINAL = 0,
    HUFFMAN = 1,
    ZSTD = 2,
    FSE = 3,
    TRUNCATE = 4,
    AUTO = 5
} MethodsEnums;

const char* getEnumName(MethodsEnums method);
#endif // METHODS_ENUM_H
 
