-- Bucket público pra upload de mídia dos fluxos (imagem/video/audio/documento).
-- Admins fazem upload via /api/admin/media/upload → retorna URL pública.
-- Bot consome URL ao executar [FLOW: nome].

insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values (
  'flow-media',
  'flow-media',
  true,
  26214400,  -- 25 MB
  array[
    'image/jpeg','image/png','image/gif','image/webp',
    'video/mp4','video/quicktime','video/webm',
    'audio/mpeg','audio/ogg','audio/wav','audio/mp4',
    'application/pdf',
    'application/msword',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
  ]
)
on conflict (id) do update set
  public = excluded.public,
  file_size_limit = excluded.file_size_limit,
  allowed_mime_types = excluded.allowed_mime_types;

-- RLS policies pro bucket
do $$
begin
  if not exists (select 1 from pg_policies where tablename='objects' and schemaname='storage' and policyname='flow_media_service_all') then
    create policy "flow_media_service_all"
      on storage.objects for all
      to service_role
      using (bucket_id = 'flow-media') with check (bucket_id = 'flow-media');
  end if;
  if not exists (select 1 from pg_policies where tablename='objects' and schemaname='storage' and policyname='flow_media_public_read') then
    create policy "flow_media_public_read"
      on storage.objects for select
      to public
      using (bucket_id = 'flow-media');
  end if;
end $$;
