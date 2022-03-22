import sys
from setuptools import setup, Extension


extra_compile_args = ["-std=c99"]
if sys.platform == "darwin":
    extra_compile_args.append("-mmacosx-version-min=10.9")
    print("Using macOS clang args")
ext_modules = [
    Extension(
        "remora.encoded_kmers",
        sources=["src/remora/encoded_kmers.pyx"],
        extra_compile_args=extra_compile_args,
        language="c",
    ),
    Extension(
        "remora.refine_signal_map_core",
        sources=["src/remora/refine_signal_map_core.pyx"],
        extra_compile_args=extra_compile_args,
        language="c",
    ),
]


if __name__ == "__main__":
    setup(
        use_pyscaffold=True,
        setup_requires=[
            "setuptools>=38.3",
            "cython",
        ],
        ext_modules=ext_modules,
    )
