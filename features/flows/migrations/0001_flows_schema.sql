-- Tabela de flows persistentes.
-- Substitui FLOW_REGISTRY in-memory quando rodando em produção multi-instance.

create table if not exists public.flows (
  id          uuid primary key default gen_random_uuid(),
  project_id  text not null,
  name        text not null,
  description text not null default '',
  steps       jsonb not null default '[]'::jsonb,
  enabled     boolean not null default true,
  created_at  timestamptz not null default now(),
  updated_at  timestamptz not null default now(),
  unique (project_id, name)
);
create index if not exists flows_project_idx on public.flows (project_id);

-- Trigger pra updated_at automático (se ainda não existir helper)
create or replace function public.touch_updated_at()
returns trigger as $$
begin
  new.updated_at = now();
  return new;
end;
$$ language plpgsql;

drop trigger if exists flows_touch on public.flows;
create trigger flows_touch before update on public.flows
  for each row execute function public.touch_updated_at();

-- RLS — só service_role acessa (bot + painel admin)
alter table public.flows enable row level security;

do $$
begin
  if not exists (
    select 1 from pg_policies
    where tablename='flows' and policyname='service_all'
  ) then
    create policy service_all on public.flows
      for all to service_role using (true) with check (true);
  end if;
end $$;
