import os
import sys
import argparse
import subprocess
import zipnn
from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)
from zipnn_decompress_file import (
    decompress_file,
)

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
        )
    )
)


def check_and_install_zipnn():
    try:
        import zipnn
    except ImportError:
        print("zipnn not found. Installing...")
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "zipnn",
            ]
        )
        import zipnn


def decompress_zpn_files(
    dtype="",
    path=".",
    delete=False,
    force=False,
    max_processes=1,
):

    file_list = []
    directories_to_search = [
        (
            path,
            [],
            os.listdir(path),
        )
    ]
    for (
        root,
        _,
        files,
    ) in directories_to_search:
        for file_name in files:
            if file_name.endswith(".zpn"):
                decompressed_path = file_name[:-4]
                if not force and os.path.exists(
                    decompressed_path
                ):
                    user_input = (
                        input(
                            f"{decompressed_path} already exists; overwrite (y/n)? "
                        )
                        .strip()
                        .lower()
                    )
                    if user_input not in (
                        "y",
                        "yes",
                    ):
                        print(
                            f"Skipping {file_name}..."
                        )
                        continue
                full_path = os.path.join(
                    root,
                    file_name,
                )
                file_list.append(full_path)

    with ProcessPoolExecutor(
        max_workers=max_processes
    ) as executor:
        for file in file_list[:max_processes]:
            future_to_file = {
                executor.submit(
                    decompress_file,
                    file,
                    dtype,
                    delete,
                    True,
                ): file
                for file in file_list[
                    :max_processes
                ]
            }

            file_list = file_list[max_processes:]
            while future_to_file:

                for future in as_completed(
                    future_to_file
                ):
                    file = future_to_file.pop(
                        future
                    )
                    try:
                        future.result()
                    except Exception as exc:
                        print(
                            f"File {file} generated an exception: {exc}"
                        )

                    if file_list:
                        next_file = file_list.pop(
                            0
                        )
                        future_to_file[
                            executor.submit(
                                decompress_file,
                                next_file,
                                dtype,
                                delete,
                                True,
                            )
                        ] = next_file
                        #


if __name__ == "__main__":
    check_and_install_zipnn()

    parser = argparse.ArgumentParser(
        description="Compresses all .zpn files. (optional) dtype."
    )
    parser.add_argument(
        "--float32",
        action="store_true",
        help="A flag that triggers float32 compression.",
    )
    parser.add_argument(
        "--path",
        type=str,
        help="Path to folder of files to decompress. If left empty, checks current folder.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="A flag that triggers deletion of a single compressed file instead of decompression",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="A flag that forces overwriting when decompressing.",
    )
    parser.add_argument(
        "--max_processes",
        type=int,
        help="The amount of maximum processes.",
    )
    args = parser.parse_args()
    optional_kwargs = {}
    if args.float32:
        optional_kwargs["dtype"] = 32
    if args.path is not None:
        optional_kwargs["path"] = args.path
    if args.delete:
        optional_kwargs["delete"] = args.delete
    if args.force:
        optional_kwargs["force"] = args.force
    if args.max_processes:
        optional_kwargs["max_processes"] = (
            args.max_processes
        )

    decompress_zpn_files(**optional_kwargs)
