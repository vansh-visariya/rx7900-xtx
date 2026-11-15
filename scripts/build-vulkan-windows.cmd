@echo off
rem Windows build (MSVC + Vulkan SDK) of the Vulkan backend. See docs/rdna3/WINDOWS.md.
rem Requires Visual Studio with the C++ toolchain and the Vulkan SDK (VULKAN_SDK set by its installer).
rem Edit VCVARS below if Visual Studio is installed elsewhere.
setlocal
if not defined VCVARS set "VCVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" (
    echo vcvars64.bat not found: set VCVARS to its path
    exit /b 1
)
if not defined VULKAN_SDK (
    echo VULKAN_SDK is not set: install the Vulkan SDK
    exit /b 1
)
call "%VCVARS%" >nul || exit /b 1
set "PATH=%VULKAN_SDK%\Bin;%PATH%"
cd /d "%~dp0.." || exit /b 1
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF || exit /b 2
cmake --build build --config Release --target llama-server test-backend-ops llama-bench || exit /b 3
echo BUILD_OK
