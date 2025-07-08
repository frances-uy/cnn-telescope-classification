#!/bin/bash
base_dir="/home/frances.uy/downloaded_files"
camera_url="http://mkoccamdev-lv1:8080/camera/100000006aa8722a"
today=$(date +%Y-%m-%d)
mkdir -p "$base_dir/$today"
wget -O "$base_dir/$today/camera.html" "$camera_url"
grep -oE '<a href=[^>]+>' "$base_dir/$today/camera.html" | sed -e 's/<a href=//g' -e 's/>//g' -e 's/^/http:\/\/mkoccamdev-lv1:8080/g' > "$base_dir/$today/image_urls.txt"
if [ -s "$base_dir/$today/image_urls.txt" ]; then
  grep -E '(1[9]|2[0-3]|0[0-5])[0-5][0-9][0-5][0-9]\.png$' "$base_dir/$today/image_urls.txt" > "$base_dir/$today/filtered_image_urls.txt"
  if [ -s "$base_dir/$today/filtered_image_urls.txt" ]; then
    wget -nc -P "$base_dir/$today" -i "$base_dir/$today/filtered_image_urls.txt"
  else
    echo "No image URLs found within the specified time range in $base_dir/$today/image_urls.txt"
  fi
else
  echo "No image URLs found in $base_dir/$today/image_urls.txt"
fi
