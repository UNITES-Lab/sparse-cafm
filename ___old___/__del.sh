cd /playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/data/mos2-cafm-controlnet-synthetic-dataset/train/images

for file in *.png; do
  # Strip out any non-numeric characters (like leading zeros or the ".npy" extension)
  num=$(echo "$file" | sed 's/[^0-9]//g')

  # Compare the numeric part to 2500 and remove if it's greater
  if [ "$num" -gt 2500 ]; then
    echo "Removing: $file"
    rm "$file"
  fi
done