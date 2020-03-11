#!/bin/bash

#Build Script

#One common use of bash scripts is for releasing a “build” of your source code. 
#Sometimes your private source code may contain developer resources or private 
#information that you don’t want to release in the published version.

#In this project, you’ll create a release script to copy certain files from a 
#source directory into a build directory.


version=1
echo "Welcome in this program"

firstline=$(head -n 1 source/changelog.md)
echo First line content of "changelog" is $firstline

read -a splitfirstline <<< $firstline
echo "Your version is ${splitfirstline[1]}"

echo "enter 1 (for yes) to continue and 0 (for no) to exit."
read versioncontinue
echo "versioncontinue value is ${versioncontinue}"

if [ $versioncontinue -eq 1 ]
then
  echo "Content of Source folder"
  for filename in source/*
  do
    if [ "$filename" == "source/secretinfo.md" ]
    then
      echo "${filename} will not being copied "
    else
      echo "${filename} will being copied "
      cp $filename build/
    fi
  done
  pwd
  cd build/
  pwd
  echo "Build version $version contains:"
  ls
  cd ..
else
  echo "Stop"
fi
