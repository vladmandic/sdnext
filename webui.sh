#!/usr/bin/env bash
# -------------------------------------------------------------------------------------------------------------
# Do not make any changes to this file, change the variables in webui-user.sh instead and call this file
# -------------------------------------------------------------------------------------------------------------

# change to local directory
cd -- "$(dirname -- "$0")"

can_run_as_root=0
export ERROR_REPORTING=FALSE
export PIP_IGNORE_INSTALLED=0

# Read variables from webui-user.sh
if [[ -f webui-user.sh ]]
then
    source ./webui-user.sh
fi

# python3 executable
PYTHON_ENV="${PYTHON}"
if [[ -z "${PYTHON}" ]]
then
    PYTHON="python3"
fi

# git executable
if [[ -z "${GIT}" ]]
then
    export GIT="git"
fi

if [[ -z "${venv_dir}" ]]
then
    venv_dir="venv"
fi

# read any command line flags to the webui.sh script
while getopts "f" flag > /dev/null 2>&1
do
    case ${flag} in
        f) can_run_as_root=1;;
        *) break;;
    esac
done

# Do not run as root unless inside a Docker container
if [[ $(id -u) -eq 0 && can_run_as_root -eq 0 && ! -f /.dockerenv ]]
then
    echo "Cannot run as root"
    exit 1
fi

for preq in "${GIT}" "${PYTHON}"
do
    if ! hash "${preq}" &>/dev/null
    then
        printf "Error: %s is not installed, aborting...\n" "${preq}"
        exit 1
    fi
done

if ! "${PYTHON}" -c "import venv" &>/dev/null
then
    echo "Error: python3-venv is not installed"
    exit 1
fi

if [[ ! -d "${venv_dir}" ]]
then
    echo "Create python venv"
    "${PYTHON}" -m venv "${venv_dir}"
    first_launch=1
fi

if [[ -f "${venv_dir}"/bin/activate ]]
then
    source "${venv_dir}"/bin/activate
    echo "Activate python venv: $VIRTUAL_ENV"
else
    echo "Error: Cannot activate python venv"
    exit 1
fi

# Add venv lib folder to PATH
if [ -d "$(realpath "$venv_dir")/lib/" ] && [[ -z "${DISABLE_VENV_LIBS}" ]]
then
    if [ -z "${LD_LIBRARY_PATH+x}" ]
    then
      export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(realpath "$venv_dir")/lib/
    else
      export LD_LIBRARY_PATH=$(realpath "$venv_dir")/lib/
    fi
fi

# Add ROCm to PATH if it's not already
if  [ ! -x "$(command -v rocminfo)" ] && [ -f '/opt/rocm/bin/rocminfo' ]
then
    export PATH=$PATH:/opt/rocm/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib
fi

if [[ -n "${ACCELERATE}" ]] && [[ "${ACCELERATE}" == "True" ]] && [ -x "$(command -v accelerate)" ]
then
    echo "Launch: accelerate"
    exec accelerate launch --num_cpu_threads_per_process=6 launch.py "$@"
elif [[ -n "${IPEXRUN}" ]] && [[ "${IPEXRUN}" == "True" ]] && [ -x "$(command -v ipexrun)" ]
then
    echo "Launch: ipexrun"
    exec ipexrun --multi-task-manager 'taskset' --memory-allocator 'jemalloc' launch.py "$@"
elif [[ -f "${venv_dir}/bin/python3" ]]
then
    PYTHON="${venv_dir}/bin/python3"
    echo "Launch: ${PYTHON}"
    exec "${PYTHON}" launch.py "$@"
else
    echo "Launch: ${PYTHON}"
    exec "${PYTHON}" launch.py "$@"
fi
