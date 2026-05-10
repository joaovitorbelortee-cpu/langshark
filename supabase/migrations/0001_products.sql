-- Migration: catálogo de produtos por projeto
-- Aplique via: supabase db push  (ou cole no SQL Editor do dashboard)

create table if not exists public.products (
  id            text primary key,
  project_id    text not null default 'padrao',
  name          text not null,
  description   text not null default '',
  price         numeric,
  metadata      jsonb not null default '{}'::jsonb,
  created_at    timestamptz not null default now(),
  updated_at    timestamptz not null default now()
);

create index if not exists products_project_idx on public.products (project_id);

-- Tabela auxiliar pra mapear instance_name → project_id
-- (compat com schema do bot antigo: instance_projects)
create table if not exists public.instance_projects (
  instance_name text primary key,
  project_id    text not null,
  created_at    timestamptz not null default now()
);

create index if not exists instance_projects_project_idx on public.instance_projects (project_id);

-- RLS: leitura pelo service role apenas (default safe).
-- O bot usa SUPABASE_SERVICE_KEY → bypass RLS automático.
alter table public.products enable row level security;
alter table public.instance_projects enable row level security;

-- Policy explícita pra service role (não obrigatória, mas explicita intent).
do $$ begin
  if not exists (select 1 from pg_policies where tablename = 'products' and policyname = 'service_all') then
    create policy service_all on public.products
      for all to service_role using (true) with check (true);
  end if;
  if not exists (select 1 from pg_policies where tablename = 'instance_projects' and policyname = 'service_all') then
    create policy service_all on public.instance_projects
      for all to service_role using (true) with check (true);
  end if;
end $$;
