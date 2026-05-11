-- Toggle ON/OFF do follow-up automático por projeto.
-- Default TRUE (mantém comportamento atual). Admin pode desativar via painel
-- pra parar follow-ups sem mexer em todo o bot (is_active continua separado).
alter table public.project_config
  add column if not exists followup_enabled boolean not null default true;
