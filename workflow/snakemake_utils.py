"""
Useful parsers, loggers etc to be used with snakemake.
"""

import logging
from time import time
from datetime import timedelta, datetime


def log_the_run(name, input, output, log, params, time_used):
    logging.basicConfig(filename=log, level="INFO")
    logger = logging.getLogger(name)
    logger.info(f"SET UP: snakemake run with identifier {name}")
    logger.info(f"Input files: {input}")
    logger.info(f"Parameters: {params}")
    logger.info(f"Output files: {output}")
    logger.info(f"Log files: {log}")
    logger.info(f"Runtime of the total run: {timedelta(seconds=time_used)} hours:minutes:seconds")
    logger.info(f"This run was finished at: {datetime.fromtimestamp(time()).isoformat()}")


def find_config_parameter_value(config_file, parameter_name):
    with open(config_file, "r") as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith(parameter_name):
            return line.strip().split("=")[1]


def modify_mdrun(path_to_file, param_to_change, new_value):
    with open(path_to_file, "r") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.startswith(param_to_change):
            lines[i] = f"{param_to_change} = {new_value}\n"
    with open(path_to_file, "w") as f:
        f.writelines(lines)


def modify_topology(path_to_file, i, j, funct, low, up1, up2, force_constant):
    with open(path_to_file,"r") as f:
        lines = f.readlines()
    for k, line in enumerate(lines):
        if line.startswith("[ angles ]"):
            break
        split_line = line.strip().split()
        if len(split_line) >= 2 and split_line[0] == i and split_line[1] == j:
            lines[k] = f"{i}\t{j}\t{funct}\t{low}\t{up1}\t{up2}\t{force_constant}\n"
    with open(path_to_file,"w") as f:
        f.writelines(lines)