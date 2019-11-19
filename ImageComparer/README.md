# Usage
`python ./imageComparer [image-directory] [minimum similarity required]`

Goes through directory of images, orders them in sequence based on frame number in name. For each frame, discards neighbouring subsequent frames that are at least **minimum similarity required** similar. Note this value is between 0 and 1 (invlusive).