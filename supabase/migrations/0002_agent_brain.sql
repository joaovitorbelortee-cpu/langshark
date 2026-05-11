-- Agent brain — system prompt particionado em seções editáveis via painel.
-- Aplique via SQL Editor Supabase ou supabase db push.

-- ============================================================
-- 1) project_config — config geral por projeto (multi-tenant)
-- ============================================================

create table if not exists public.project_config (
  project_id      text primary key,
  display_name    text not null default '',
  agent_name      text not null default '',
  ai_model        text not null default 'openai/gpt-4o-mini',
  ai_temperature  numeric(3,2) not null default 0.7 check (ai_temperature between 0 and 2),
  ai_max_tokens   int not null default 600 check (ai_max_tokens between 50 and 8000),
  is_active       boolean not null default true,
  brain_sections  jsonb not null default '{
    "company_info":         {"content": "", "max_chars": 7000, "icon": "building",  "title": "Informacoes da Empresa"},
    "prices":               {"content": "", "max_chars": 7000, "icon": "dollar",    "title": "Precos e Valores"},
    "parameters":           {"content": "", "max_chars": 7000, "icon": "settings",  "title": "Parametros"},
    "priority_situations":  {"content": "", "max_chars": 7000, "icon": "alert",     "title": "Situacoes Prioritarias"},
    "knowledge_base":       {"content": "", "max_chars": 7000, "icon": "book",      "title": "Base de Conhecimento"}
  }'::jsonb,
  api_keys        jsonb not null default '{}'::jsonb,
  created_at      timestamptz not null default now(),
  updated_at      timestamptz not null default now()
);

create or replace function public.touch_updated_at() returns trigger as $$
begin
  new.updated_at = now();
  return new;
end;
$$ language plpgsql;

drop trigger if exists project_config_touch on public.project_config;
create trigger project_config_touch
  before update on public.project_config
  for each row execute function public.touch_updated_at();

-- ============================================================
-- 2) admin_users — autenticação do painel
-- ============================================================

create table if not exists public.admin_users (
  id            uuid primary key default gen_random_uuid(),
  email         text not null unique,
  password_hash text not null,
  display_name  text,
  project_ids   text[] not null default '{}'::text[],
  created_at    timestamptz not null default now(),
  last_login_at timestamptz
);

-- ============================================================
-- 3) ai_models_catalog — catálogo curado de modelos pra dropdown
-- ============================================================

create table if not exists public.ai_models_catalog (
  model_id        text primary key,
  provider        text not null,
  display_name    text not null,
  input_price     numeric(10,4),
  output_price    numeric(10,4),
  tier            text not null default 'standard',
  supports_vision boolean not null default false,
  active          boolean not null default true,
  sort_order      int not null default 0
);

insert into public.ai_models_catalog
  (model_id, provider, display_name, input_price, output_price, tier, supports_vision, sort_order) values
  ('openai/gpt-4o-mini',          'openai',    'GPT-4o mini',         0.15,  0.60,  'budget',   true,  10),
  ('openai/gpt-4o',               'openai',    'GPT-4o',              2.50, 10.00,  'premium',  true,  20),
  ('anthropic/claude-haiku-4.5',  'anthropic', 'Claude Haiku 4.5',    1.00,  5.00,  'standard', true,  30),
  ('anthropic/claude-sonnet-4.5', 'anthropic', 'Claude Sonnet 4.5',   3.00, 15.00,  'premium',  true,  40),
  ('google/gemini-2.5-flash',     'google',    'Gemini 2.5 Flash',    0.075, 0.30,  'budget',   true,  50),
  ('google/gemini-2.5-pro',       'google',    'Gemini 2.5 Pro',      1.25,  5.00,  'premium',  true,  60),
  ('deepseek/deepseek-chat',      'deepseek',  'DeepSeek V3',         0.27,  1.10,  'budget',   false, 70),
  ('google/gemma-3-27b-it:free',  'google',    'Gemma 3 27B (free)',  0.00,  0.00,  'free',     true,  80)
on conflict (model_id) do update set
  display_name=excluded.display_name,
  input_price=excluded.input_price,
  output_price=excluded.output_price,
  tier=excluded.tier,
  supports_vision=excluded.supports_vision,
  sort_order=excluded.sort_order;

-- ============================================================
-- 4) RLS — service role bypass (painel usa SERVICE_KEY)
-- ============================================================

alter table public.project_config    enable row level security;
alter table public.admin_users       enable row level security;
alter table public.ai_models_catalog enable row level security;

do $$
declare
  r record;
begin
  for r in select tablename from pg_tables where schemaname='public'
           and tablename in ('project_config','admin_users','ai_models_catalog')
  loop
    if not exists (select 1 from pg_policies where tablename=r.tablename and policyname='service_all') then
      execute format(
        'create policy service_all on public.%I for all to service_role using (true) with check (true)',
        r.tablename
      );
    end if;
  end loop;
end $$;

-- ============================================================
-- 5) Seed projeto padrao (vazio — popular via scripts/seed_brain.py)
-- ============================================================

insert into public.project_config (project_id, display_name, agent_name)
values ('padrao', 'Meu Primeiro Projeto', 'João')
on conflict (project_id) do nothing;
