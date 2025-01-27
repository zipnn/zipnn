import os
import sys
import argparse
from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)
from zipnn_decompress_file import (
    decompress_file,
)
import multiprocessing
from util import check_and_install_zipnn, RED, RESET, GREEN, YELLOW

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
        )
    )
)

def replace_in_file(file_path, old: str, new: str) -> None:
    """Given a file_path, replace all occurrences of `old` with `new` inpalce."""

    with open(file_path, 'r') as file:
        file_data = file.read()

    file_data = file_data.replace(old, new)

    with open(file_path, 'w') as file:
        file.write(file_data)

def decompress_znn_files(
    path=".",
    delete=False,
    force=False,
    max_processes=1,
    hf_cache=False,
    model="",
    branch="main",
    threads=multiprocessing.cpu_count()
):
    import zipnn

    overwrite_first=True

    if model:
        if not hf_cache:
            raise ValueError(
                "Must specify --hf_cache when using --model"
            )
        try:
            from huggingface_hub import scan_cache_dir
        except ImportError:
            raise ImportError(
                "huggingface_hub not found. Please pip install huggingface_hub."
            )  
        cache = scan_cache_dir()
        repo = next((repo for repo in cache.repos if repo.repo_id == model), None)

        if repo is not None:
            print(f"Found repo {model} in cache")
            
            # Get the latest revision path
            hash = ''
            try:
                with open(os.path.join(repo.repo_path, 'refs', branch), "r") as ref:
                    hash = ref.read()
            except FileNotFoundError:
                raise FileNotFoundError(f"Branch {branch} not found in repo {model}")
            
            path = os.path.join(repo.repo_path, 'snapshots', hash)

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
            if file_name.endswith(".znn"):
                decompressed_path = file_name[:-4]
                if not force and os.path.exists(
                    decompressed_path
                ):
                    #
                    if overwrite_first:
                        overwrite_first=False
                        user_input = (
                            input(
                                f"Decompressed files already exists; Would you like to overwrite them all (y/n)? "
                            )
                            .strip()
                            .lower()
                        )
                        if user_input not in (
                            "y",
                            "yes",
                        ):
                            print(
                                f"No forced overwriting."
                            )
                        else:
                            print(
                                f"Overwriting all decompressed files."
                            )
                            force=True
                        
                    #
                    if not force:
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

    if file_list and hf_cache:
        try:
            from transformers.utils import (
                SAFE_WEIGHTS_INDEX_NAME,
                WEIGHTS_INDEX_NAME
            )
        except ImportError:
            raise ImportError(
                "Transformers not found. Please pip install transformers."
            )
            
        suffix = file_list[0].split('/')[-1].split('.')[-2] # get the one before .znn

        if os.path.exists(os.path.join(path, SAFE_WEIGHTS_INDEX_NAME)):
            print(f"{YELLOW}Fixing Hugging Face model json...{RESET}")
            blob_name = os.path.join(path, os.readlink(os.path.join(path, SAFE_WEIGHTS_INDEX_NAME)))
            replace_in_file(
                    file_path=blob_name,
                    old=f"{suffix}.znn",
                    new=f"{suffix}"
                )
        elif os.path.exists(os.path.join(path, WEIGHTS_INDEX_NAME)):
            print(f"{YELLOW}Fixing Hugging Face model json...{RESET}")
            blob_name = os.path.join(path, os.readlink(os.path.join(path, WEIGHTS_INDEX_NAME)))
            replace_in_file(
                    file_path=blob_name,
                    old=f"{suffix}.znn",
                    new=f"{suffix}"
                )

    with ProcessPoolExecutor(
        max_workers=max_processes
    ) as executor:
        for file in file_list[:max_processes]:
            future_to_file = {
                executor.submit(
                    decompress_file,
                    file,
                    delete,
                    True,
                    hf_cache,
                    threads,
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
                            f"{RED}File {file} generated an exception: {exc}{RESET}"
                        )

                    if file_list:
                        next_file = file_list.pop(
                            0
                        )
                        future_to_file[
                            executor.submit(
                                decompress_file,
                                next_file,
                                delete,
                                True,
                                hf_cache,
                                threads,
                            )
                        ] = next_file
                        #
    print(f"{GREEN}All files decompressed{RESET}")


if __name__ == "__main__":
    check_and_install_zipnn()

    parser = argparse.ArgumentParser(
        description="Compresses all .znn files."
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
    parser.add_argument(
        "--hf_cache",
        action="store_true",
        help="A flag that indicates if the file is in the Hugging Face cache. Must either specify --model or --path to the model's snapshot cache.",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Only when using --hf_cache, specify the model name or path. E.g. 'ibm-granite/granite-7b-instruct'",
    )
    parser.add_argument(
        "--model_branch",
        type=str,
        default="main",
        help="Only when using --model, specify the model branch. Default is 'main'",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=multiprocessing.cpu_count(),
        help="The amount of threads to be used.",
    )
    args = parser.parse_args()
    optional_kwargs = {}
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
    if args.hf_cache:
        optional_kwargs["hf_cache"] = args.hf_cache
    if args.model:
        optional_kwargs["model"] = args.model
    if args.model_branch:
        optional_kwargs[
            "branch"
        ] = args.model_branch
    if args.threads:
        optional_kwargs["threads"] = args.threads#

    decompress_znn_files(**optional_kwargs)
