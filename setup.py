from setuptools import setup, find_packages

install_requires = []
with open("./requirements.txt", "r") as requirements_file:
    # don't include gptq_llama by default
    reqs = [r.strip() for r in requirements_file.readlines() if "gptq_llama" not in r]
    for r in reqs:
        install_requires.append(r)


setup(
    name='alpaca_lora_4bit',
    version='0.1',
    description='Alpaca LoRA 4-bit',
    package_dir={'': 'src'},
    packages=find_packages(),
    install_requires=install_requires,
    extras_require={
        'cuda': 'gptq_llama @ git+https://github.com/sterlind/GPTQ-for-LLaMa.git@lora_4bit',
        'triton': 'triton',
    },
)
