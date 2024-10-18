#include "methods_enums.h"

const char* getEnumName(MethodsEnums method) {
    switch (method) {
        case ORIGINAL: return "ORIGINAL";
        case HUFFMAN: return "HUFFMAN";
        case ZSTD: return "ZSTD";
        case FSE: return "FSE";
        case TRUNCATE: return "TRUNCATE";
        case AUTO: return "AUTO";
        default: return "UNKNOWN";
    }
}

