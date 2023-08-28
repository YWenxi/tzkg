import os
from tzkg.data_processing import Transfer
from tzkg.datasets.utils import ensure_dir
import pytest

datasets_dir = [
    "./test-data",
]

datasets_name = [
    "weapons", "cdmo"
]

test_output_dir = "test_output"

name_space = "http://tzzn.kg.cn/#"

def _test_some_dataset(data_dir):
    """
    Check all saving functionals
    """

    print(f"Testing on dataset: {data_dir}")

    input_files = os.listdir(data_dir)

    try:
        save_dir = os.path.join(data_dir, test_output_dir)
        print(f"--> Saving output files to {save_dir}")
        os.mkdir(save_dir)
    except FileExistsError:
        pass

    for file in input_files:
        ext = file.split(".")[-1]
        if ext in ["csv", "owl", "rdf"]:
            print(f"------> Found input file: {file}.", end=" ")
            name = file.split(".")[-2]
        else:
            continue
        trf = Transfer(os.path.join(data_dir, file), name_space)
        print(f"Loaded.")
        
        out_name = os.path.join(save_dir, name)
        print(f"------> Saving to: {file}.")

        trf.csv_to_onto(out_name, out_format="rdf")
        trf._to_triples(out_name, "csv")
        trf._to_triples(out_name, "txt")
        trf._to_trainds(out_name, save=True, out_type="dict")

        def _print_saved_files(save_file_list):
            outs = ""
            for filename in save_file_list:
                outs += f"\t{filename}\t\t\n"
            return outs
        print(f"------> Done! Now you should have:\n{_print_saved_files(os.listdir(save_dir))}.")


@pytest.mark.dependency()
def test_dataset_saving_functionals():
    for dir in datasets_dir:
        for name in datasets_name:
            _test_some_dataset(os.path.join(dir, name))
            
            check_list = [
                f"{name}.{ext}" for ext in ["csv", "txt", "rdf"]
            ]
            check_list += [
                f"{name}_{tail}.dict" for tail in ["entity2id", "relation2id"]
            ]

            check_dir = os.path.join(dir, name, "test_output")
            current_file_list = os.listdir(check_dir)
            for check_file in check_list:
                if check_file not in current_file_list:
                    raise FileNotFoundError(f"{check_file} has not been generated in {check_dir}")
                
            # when all files are checked remove test_output directory
            os.system(f"rm -r {check_dir}")