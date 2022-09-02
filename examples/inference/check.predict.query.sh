#!/bin/bash

CSV_FILE=labels/2022_0901_1140_43.auto.label.csv

printf "Check falsely predicted single-hit...\n"
awk -F ',' 'substr($1,6,1) != 1 && $3 == 1 {print $0}' $CSV_FILE
printf "Done.\n"

printf "\n"

printf "Check falsely predicted multi-hit.\n"
awk -F ',' 'substr($1,6,1) == 1 && $3 != 1 {print $0}' $CSV_FILE
printf "Done.\n"
