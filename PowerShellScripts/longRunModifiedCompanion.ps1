$VENV_PATH = "C:\Users\AADHIL SANU\Desktop\projects\RK2 and Langevin\Scripts\Activate.ps1"


$epsilon = .1526418

$N_0_squared = 596.36


& $VENV_PATH


for ($i = 0; $i -lt 1; $i++) {
    $iterNo = $i + 1
    Write-Host "Running simulation iteration = $iterNo"
    python longRunModified.py --epsilon $epsilon --N_0_squared $N_0_squared
}


