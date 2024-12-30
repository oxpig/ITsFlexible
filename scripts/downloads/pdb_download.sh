#!/bin/bash
# Description: Downloads and unzips all pdb files

pdb_download_dir="../data/download_dir"
mkdir -p "$pdb_download_dir"

echo "Downloading PDB database"
rsync --recursive --links --perms --times --compress --info=progress2 --delete --port=33444 rsync.rcsb.org::ftp_data/structures/divided/pdb/ "$pdb_download_dir"

find "$pdb_download_dir/" -type f -name "*.gz" -exec gunzip {} \+
find "$pdb_download_dir" -type d -empty -delete

echo "Moving PDB files to ../data/PDB_structures"

mkdir -p ../data/PDB_structures

for sub_dir in "$pdb_download_dir"/*; do
 mv "$sub_dir/"*.ent ../data/PDB_structures/
done

find "$pdb_download_dir" -type d -empty -delete

for file in ../data/PDB_structures/*.ent; do
 new_file="${file%.ent}.pdb"  # Change the extension from .ent to .pdb
 mv "$file" "${new_file:3}"  # Remove the first four letters
done