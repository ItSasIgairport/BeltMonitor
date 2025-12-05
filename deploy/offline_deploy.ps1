$ErrorActionPreference = "Stop"

# Configuration
$REMOTE_HOST = "10.34.45.20"
$REMOTE_USER = "havaciliksistemleri"
$REMOTE_PASS = "21072025"
$REMOTE_DIR = "~/BeltMonitor"
$IMAGE_NAME = "belt-monitorv3"
$CONTAINER_NAME = "belt-monitor-latest-container"
$BUNDLE_NAME = "update.bundle"
$LOCAL_BUNDLE_PATH = "deploy/$BUNDLE_NAME"
$BASE_IMAGE_TAR = "belt_monitor_base.tar"
$LOCAL_BASE_TAR_PATH = "deploy/$BASE_IMAGE_TAR"

Write-Host "Step 0: Checking Pre-requisites..."
if (-not (Test-Path $LOCAL_BASE_TAR_PATH)) {
    Write-Warning "Base image tar '$LOCAL_BASE_TAR_PATH' not found locally."
    Write-Warning "You should build it first: docker build -f deploy/Dockerfile.base -t belt-monitor-base:latest ."
    Write-Warning "Then save it: docker save belt-monitor-base:latest -o $LOCAL_BASE_TAR_PATH"
    # We don't exit here just in case you know what you are doing or it's already on remote
}

Write-Host "Step 0.5: Creating Git Bundle locally..."
# Create a bundle of the 'main' branch
# Ensure artifacts directory exists
if (-not (Test-Path "deploy")) { New-Item -ItemType Directory -Path "deploy" | Out-Null }
git bundle create $LOCAL_BUNDLE_PATH main
if ($LASTEXITCODE -ne 0) { Write-Error "Failed to create git bundle"; exit 1 }

Write-Host "Step 1: Transferring Bundle to $REMOTE_HOST..."
# Use SCP to copy the bundle to the remote server
scp $LOCAL_BUNDLE_PATH "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/${BUNDLE_NAME}"
if ($LASTEXITCODE -ne 0) { Write-Error "Failed to transfer bundle via SCP"; exit 1 }


Write-Host "Connecting to $REMOTE_HOST to deploy..."

# We use a Here-String @" ... "@ to avoid quoting issues in PowerShell.
$remote_command = @"
    echo 'Step 2: Navigate to directory'
    cd $REMOTE_DIR || { echo 'Failed to change directory'; exit 1; }

    echo 'Step 3: Git Pull from Bundle'
    git pull $BUNDLE_NAME main || { echo 'Git pull from bundle failed'; exit 1; }

    if [ -f "$BASE_IMAGE_TAR" ]; then
        echo 'Step 3.5: Load Base Image from Tar'
        echo $REMOTE_PASS | sudo -S docker load -i $BASE_IMAGE_TAR || echo "Failed to load base image, continuing..."
    fi

    echo 'Step 4: Docker Build (Offline)'
    # Ensure base image is available
    if ! echo $REMOTE_PASS | sudo -S docker image inspect belt-monitor-base:latest >/dev/null 2>&1; then
         echo "WARNING: belt-monitor-base:latest not found! Build will likely fail."
    fi

    # Always build fresh from the base image using the standard Dockerfile
    # We use --network none to ensure strictly offline build
    echo $REMOTE_PASS | sudo -S docker build --network none -f deploy/Dockerfile -t ${IMAGE_NAME}:latest . || { echo 'Docker build failed'; exit 1; }

    echo 'Step 5: Save and Import Image'
    echo $REMOTE_PASS | sudo -S sh -c 'docker save ${IMAGE_NAME}:latest | ctr -n k8s.io images import -' || { echo 'Image import failed'; exit 1; }

    echo 'Step 6: Restart Deployment'
    kubectl rollout restart deployment/belt-monitor-deployment -n aviation
    kubectl rollout status deployment/belt-monitor-deployment -n aviation
    
    rm $BUNDLE_NAME
"@

# Normalize line endings to LF for Linux compatibility
# We use -replace "\r", "" to kill ALL carriage returns, ensuring clean Linux script
$remote_command = $remote_command -replace "\r", ""

# Execute via SSH
$remote_command | ssh -t "${REMOTE_USER}@${REMOTE_HOST}" "bash -s"

if ($LASTEXITCODE -eq 0) {
    Write-Host "Remote deployment completed successfully."
    Remove-Item $LOCAL_BUNDLE_PATH -ErrorAction SilentlyContinue
} else {
    Write-Error "Remote deployment failed with exit code $LASTEXITCODE."
}
