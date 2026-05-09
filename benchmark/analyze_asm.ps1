param(
    [Parameter(Mandatory = $true)]
    [string]$Object,

    [string]$Function = "subgraph_0",

    [string]$Objdump = "objdump",

    [string]$OutAsm = ""
)

if (-not (Test-Path $Object)) {
    throw "Object file not found: $Object"
}

$asm = & $Objdump -d -M intel $Object
if ($LASTEXITCODE -ne 0) {
    throw "objdump failed for $Object"
}

if ($OutAsm -ne "") {
    $asm | Set-Content -Path $OutAsm -Encoding ASCII
}

$start = -1
for ($i = 0; $i -lt $asm.Count; ++$i) {
    if ($asm[$i] -match "^([0-9a-fA-F]+) <$([regex]::Escape($Function))>:") {
        $start = $i
        break
    }
}

if ($start -lt 0) {
    throw "Function not found in object: $Function"
}

$end = $asm.Count
for ($i = $start + 1; $i -lt $asm.Count; ++$i) {
    if ($asm[$i] -match "^[0-9a-fA-F]+ <.*>:") {
        $end = $i
        break
    }
}

$body = $asm[$start..($end - 1)]

function Count-Pattern([string]$Pattern) {
    return ($body -match $Pattern).Count
}

$stats = [ordered]@{
    Object        = (Resolve-Path $Object).Path
    Function      = $Function
    Lines         = $body.Count
    PackedFMA     = Count-Pattern "vfmadd[0-9]+ps"
    ScalarFMA     = Count-Pattern "vfmadd[0-9]+ss"
    ZmmPackedFMA  = Count-Pattern "vfmadd[0-9]+ps.*zmm"
    YmmPackedFMA  = Count-Pattern "vfmadd[0-9]+ps.*ymm"
    XmmPackedFMA  = Count-Pattern "vfmadd[0-9]+ps.*xmm"
    Gather        = Count-Pattern "gather"
    Scatter       = Count-Pattern "scatter"
    StackVectorOp = Count-Pattern "v(mov|add|mul|fmadd).*\[(rsp|rbp)"
    VectorLoad    = Count-Pattern "vmovups|vmovaps"
    ScalarMove    = Count-Pattern "vmovss"
    Broadcast     = Count-Pattern "vbroadcast"
    Prefetch      = Count-Pattern "prefetch"
}

[pscustomobject]$stats | Format-List
