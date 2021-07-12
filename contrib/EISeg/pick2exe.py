from qpt.executor import CreateExecutableModule

if __name__ == "__main__":
    module = CreateExecutableModule(
        work_dir="contrib/EISeg/eiseg", 
        launcher_py_path="contrib/EISeg/eiseg/exe.py", 
        save_path="contrib/EISeg/out"
    )
    module.make()