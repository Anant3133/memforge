/**
 * Known defects, so a sanitizer's clean verdict can be falsified.
 *
 * `scripts/cpp_asan.py` compiles and runs this before reporting anything about
 * the real code. A sanitizer that is misconfigured, or whose runtime DLL is
 * missing, detects nothing -- which is indistinguishable from code that has
 * nothing to detect. This file makes that distinction observable: each mode
 * contains exactly one defect of a known class, and the runner reports which
 * classes each pass actually caught.
 *
 * Deliberately not a test: it is *supposed* to fail under a working sanitizer,
 * so it must never be linked into the normal test suites.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef MEMFORGE_CRT_LEAK_CHECK
#include <crtdbg.h>
#endif

int main(int argc, char** argv) {
#ifdef MEMFORGE_CRT_LEAK_CHECK
    // Must match what the real driver does, or the capability check measures a
    // different configuration than the one whose results are being trusted.
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
    _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_WARN, _CRTDBG_FILE_STDERR);
#endif

    const char* mode = argc > 1 ? argv[1] : "clean";

    if (std::strcmp(mode, "leak") == 0) {
        // Never freed. LeakSanitizer / CRT debug heap territory.
        void* p = std::malloc(1024);
        std::memset(p, 0xAB, 1024);
        std::printf("abandoned 1024 bytes at %p\n", p);
        return 0;
    }

    if (std::strcmp(mode, "uaf") == 0) {
        int* p = static_cast<int*>(std::malloc(sizeof(int) * 4));
        p[0] = 1;
        std::free(p);
        p[1] = 2;  // use after free
        std::printf("wrote through a freed pointer: %d\n", p[1]);
        return 0;
    }

    if (std::strcmp(mode, "overflow") == 0) {
        int* p = static_cast<int*>(std::malloc(sizeof(int) * 4));
        p[7] = 99;  // heap buffer overflow
        std::printf("wrote past the end: %d\n", p[7]);
        std::free(p);
        return 0;
    }

    std::printf("clean run\n");
    return 0;
}
