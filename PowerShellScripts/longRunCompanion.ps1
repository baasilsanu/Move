$VENV_PATH = "C:\Users\AADHIL SANU\Desktop\projects\RK2 and Langevin\Scripts\Activate.ps1"


$epsilon = 0.12394270273516043

$N_0_squared = 318.8640217310387

$baseFilePath = "C:\Users\AADHIL SANU\Desktop\projects\RK2 and Langevin\powerShellOut\"
$fileName = $args[0] + "_epsilon_${epsilon}_N0squared_${N_0_squared}.txt"
$outPutFilePath = Join-Path $baseFilePath -ChildPath $fileName

& $VENV_PATH


for ($i = 0; $i -lt 2500; $i++) {
    $iterNo = $i + 1
    Write-Host "Running simulation iteration = $iterNo"
    python longRun.py --epsilon $epsilon --N_0_squared $N_0_squared --Count $iterNo
}


