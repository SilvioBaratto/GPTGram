import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='WhatsGPT',
    version='0.1.0',
    author=['Silvio Baratto', 'Valeria Insogna'],
    author_email=['SILVIOANGELO.BARATTOROLDAN@studenti.units.it', 'VALERIA.INSOGNA@studenti.units.it'],
    description='Whatsapp chatbot API using generative pretraining transformers',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SilvioBaratto/WhatsGPT",
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        "Operating System :: OS Independent", 
        'Programming Language :: Python :: 3',
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=[
        "torch>=1.9.1",
        "numpy>=1.21.2",
    ],
)
