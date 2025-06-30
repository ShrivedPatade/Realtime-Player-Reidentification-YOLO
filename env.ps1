param (
    [string]$action
)

$envPath = "D:\GitHub\Realtime-Player-Reidentification-YOLO-DeepSORT\env\Scripts"

switch ($action) {
    'a' {
        & "$envPath\Activate.ps1"
    }
    'd' {
        if ($env:VIRTUAL_ENV) {
            deactivate
        } else {
            Write-Host "No virtual environment is currently active."
        }
    }
    default {
        Write-Host "Usage:"
        Write-Host "  .\env.ps1 a   # activate"
        Write-Host "  .\env.ps1 d   # deactivate"
    }
}
