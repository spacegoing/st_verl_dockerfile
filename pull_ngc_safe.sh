#!/bin/bash

# --- CONFIGURATION ---

# 1. The Target Registry (Ensure it ends with a slash '/')
TARGET_REGISTRY="registry.cn-hangzhou.aliyuncs.com/spacegoing/"

# 2. The List of Source Images
# "nvcr.io/nvidia/nemo:25.11.01"
# "nvcr.io/nvidia/nemo:25.11"
# "nvcr.io/nvidia/nemo:25.09"
# "nvcr.io/nvidia/nemo:25.09.02"
# "nvcr.io/nvidia/nemo:25.09.01"
# "nvcr.io/nvidia/nemo:25.07.gpt_oss"

IMAGES=(
    "nvcr.io/nvidia/25.12-py3"
    "nvcr.io/nvidia/25.11-py3"
    "nvcr.io/nvidia/25.10-py3"
    "nvcr.io/nvidia/25.09-py3"
)

# Optional: Set to true to delete images from local disk after push to save space
FREE_SPACE_AFTER_SUCCESS=false

# ---------------------

# Track status (0=pending, 1=success)
declare -A SYNC_STATUS
for img in "${IMAGES[@]}"; do
    SYNC_STATUS["$img"]=0
done

echo "Starting Sync: NVCR -> Aliyun"
echo "Target: $TARGET_REGISTRY"
echo "---------------------------------------------------------"

while true; do
    PENDING_COUNT=0

    for source_img in "${IMAGES[@]}"; do
        if [ "${SYNC_STATUS["$source_img"]}" -eq 0 ]; then
            
            # 1. Determine Target Name
            # ${source_img##*/} takes everything after the last '/', yielding "nemo:25.xx"
            image_basename="${source_img##*/}"
            target_img="${TARGET_REGISTRY}${image_basename}"

            echo -e "\n[Processing] $image_basename"

            # 2. PULL
            echo "   -> Pulling from NVCR..."
            if docker pull "$source_img"; then
                
                # 3. TAG
                echo "   -> Tagging as $target_img..."
                docker tag "$source_img" "$target_img"

                # 4. PUSH
                echo "   -> Pushing to Target..."
                if docker push "$target_img"; then
                    echo "   [SUCCESS] Synced $image_basename"
                    SYNC_STATUS["$source_img"]=1
                    
                    # 5. OPTIONAL CLEANUP
                    if [ "$FREE_SPACE_AFTER_SUCCESS" = true ]; then
                        echo "   -> Cleaning up local images to save space..."
                        docker rmi "$target_img" "$source_img" 2>/dev/null
                    fi
                else
                    echo "   [FAIL] Push failed. Will retry next cycle."
                    PENDING_COUNT=$((PENDING_COUNT + 1))
                fi
            else
                echo "   [FAIL] Pull failed. Will retry next cycle."
                PENDING_COUNT=$((PENDING_COUNT + 1))
            fi
        fi
    done

    # Final Check
    if [ "$PENDING_COUNT" -eq 0 ]; then
        echo -e "\n---------------------------------------------------------"
        echo "All images synced successfully!"
        break
    else
        echo -e "\n---------------------------------------------------------"
        echo "$PENDING_COUNT image(s) failed or incomplete. Retrying in 10 seconds..."
        sleep 10
    fi
done
