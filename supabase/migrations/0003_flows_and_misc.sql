-- Flows persistentes (substitui in-memory FLOW_REGISTRY de agent/flows.py)
-- Conversation tags (F4 — labels manuais admin)
-- Audit log (F4 minimal)

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

drop trigger if exists flows_touch on public.flows;
create trigger flows_touch before update on public.flows
  for each row execute function public.touch_updated_at();

create table if not exists public.audit_log (
  id            bigserial primary key,
  admin_id      uuid,
  admin_email   text,
  action        text not null,
  resource_type text not null,
  resource_id   text,
  before_state  jsonb,
  after_state   jsonb,
  created_at    timestamptz not null default now()
);
create index if not exists audit_log_resource_idx on public.audit_log (resource_type, resource_id);

alter table public.flows     enable row level security;
alter table public.audit_log enable row level security;

do $$
declare
  r record;
begin
  for r in select tablename from pg_tables where schemaname='public'
           and tablename in ('flows','audit_log') loop
    if not exists (select 1 from pg_policies where tablename=r.tablename and policyname='service_all') then
      execute format('create policy service_all on public.%I for all to service_role using (true) with check (true)', r.tablename);
    end if;
  end loop;
end $$;
