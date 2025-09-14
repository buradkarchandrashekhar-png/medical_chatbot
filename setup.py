from setuptools import setup, find_packages

setup(
    name="medical_chatbot",
    version="0.1.0",
    description="Medical chatbot project",
    author="chandra",
    author_email="chandraburadkar@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[],
    python_requires=">=3.9",
)