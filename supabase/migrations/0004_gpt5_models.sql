-- Adiciona linha GPT-5 (lançada 2025) + GPT-5.4 (lançada Mar 2026) + GPT-5.5 (Abr 2026)
-- ao catálogo de modelos. Idempotente via ON CONFLICT.
--
-- Pricing fontes:
--   - openrouter.ai/openai/gpt-5 (Apr 2026 snapshot)
--   - openrouter.ai/openai/gpt-5.4-nano + gpt-5.4-mini
--   - openrouter.ai/openai/gpt-5.5
--   - openai.com/index/introducing-gpt-5-4-mini-and-nano (Mar 17, 2026)

insert into public.ai_models_catalog
  (model_id, provider, display_name, input_price, output_price, tier, supports_vision, sort_order)
values
  -- Linha GPT-5 base
  ('openai/gpt-5-nano',      'openai', 'GPT-5 nano',      0.05,   0.40,  'budget',   false, 5),
  ('openai/gpt-5-mini',      'openai', 'GPT-5 mini',      0.25,   2.00,  'standard', true,  6),
  ('openai/gpt-5',           'openai', 'GPT-5',           1.25,   10.00, 'premium',  true,  7),

  -- Linha GPT-5.4 (Mar 17, 2026)
  ('openai/gpt-5.4-nano',    'openai', 'GPT-5.4 nano',    0.20,   1.25,  'budget',   false, 8),
  ('openai/gpt-5.4-mini',    'openai', 'GPT-5.4 mini',    0.75,   4.50,  'standard', true,  9),

  -- GPT-5.5 flagship (Abr 24, 2026)
  ('openai/gpt-5.5',         'openai', 'GPT-5.5',         5.00,  30.00,  'premium',  true,  11)
on conflict (model_id) do update set
  display_name    = excluded.display_name,
  input_price     = excluded.input_price,
  output_price    = excluded.output_price,
  tier            = excluded.tier,
  supports_vision = excluded.supports_vision,
  sort_order      = excluded.sort_order,
  active          = true;
