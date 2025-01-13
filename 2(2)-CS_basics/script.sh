#!/bin/bash

if [ ! -d "$HOME/miniconda" ]; then
    echo "Miniconda가 설치되어 있지 않습니다. 설치를 진행합니다."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O Miniconda3.sh
    bash Miniconda3.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
else
    echo "Miniconda가 이미 설치되어 있습니다. 설치를 건너뜁니다."
fi

source ~/miniconda/bin/activate

if ! python -c "import mypy" &> /dev/null; then
    echo "Mypy가 설치되어 있지 않습니다. 설치를 진행합니다."
    conda install -y mypy
else
    echo "Mypy가 이미 설치되어 있습니다. 설치를 건너뜁니다."
fi

input_dir="$HOME/input"
output_dir="$HOME/output"
submission_dir="$HOME/submission"

for file in $submission_dir/*.py; do
    filename=$(basename "$file" .py)
    input_file="$input_dir/${filename}_input"
    output_file="$output_dir/${filename}_output"

    python $file < $input_file > $output_file
done

mypy $submission_dir/*.py