# scripts/railway_setvars.ps1
# Setta todas as variáveis do Railway de uma vez.
# Antes de rodar: edite os valores abaixo OU use um .env e carregue.
#
# Uso:
#   .\scripts\railway_setvars.ps1
#
# Pré-requisito: railway login + railway link (rodar uma vez no projeto certo).

$ErrorActionPreference = "Stop"

# Tente carregar de .env.production se existir; senão use defaults inline (EDITE!)
$envFile = ".env.production"
$vars = @{}

if (Test-Path $envFile) {
    Write-Host "[setvars] Lendo $envFile..."
    Get-Content $envFile | ForEach-Object {
        if ($_ -match "^\s*#") { return }
        if ($_ -match "^\s*$") { return }
        if ($_ -match "^([A-Z_]+)\s*=\s*(.*)$") {
            $vars[$matches[1]] = $matches[2].Trim('"').Trim("'")
        }
    }
} else {
    Write-Host "[setvars] $envFile não encontrado — usando defaults inline (EDITE este script)."
    $vars = @{
        # === auth webhook ===
        "WEBHOOK_SECRET"            = "TROQUE-PARA-32-CHARS-FORTES"

        # === Evolution ===
        "EVOLUTION_API_URL"         = "https://sua-evo.example.com"
        "EVOLUTION_API_KEY"         = ""
        "EVOLUTION_INSTANCE"        = "botzap"

        # === LLM ===
        "OPENROUTER_API_KEY"        = "sk-or-..."
        "AI_MODEL"                  = "openai/gpt-4o-mini"
        "AI_BASE_URL"               = "https://openrouter.ai/api/v1"

        # === Redis ===
        "UPSTASH_REDIS_REST_URL"    = ""
        "UPSTASH_REDIS_REST_TOKEN"  = ""

        # === QStash ===
        "QSTASH_TOKEN"              = ""
        "QSTASH_URL"                = "https://qstash.upstash.io"
        "PUBLIC_BASE_URL"           = ""    # preencha após railway domain

        # === Supabase ===
        "SUPABASE_URL"              = ""
        "SUPABASE_SERVICE_KEY"      = ""
        "POSTGRES_URL"              = ""

        # === multi-tenant default ===
        "DEFAULT_PROJECT_ID"        = "padrao"

        # === RAG (já default no Dockerfile) ===
        "CHROMA_DIR"                = "/data/chroma"
    }
}

Write-Host "[setvars] Setting $($vars.Count) variables..."
$skipped = 0
foreach ($k in $vars.Keys) {
    $v = $vars[$k]
    if ([string]::IsNullOrWhiteSpace($v) -or $v -like "*TROQUE*" -or $v -like "*example*" -or $v -like "*..." ) {
        Write-Host "  [SKIP] $k (valor placeholder/vazio)"
        $skipped++
        continue
    }
    Write-Host "  [SET]  $k"
    railway variables --set "$k=$v" | Out-Null
}

Write-Host ""
Write-Host "[setvars] Done. Skipped $skipped placeholders."
Write-Host "[setvars] Próximo: railway redeploy"
