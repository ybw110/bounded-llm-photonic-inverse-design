# Spec Parser Prompt

You convert a metasurface design request into strict JSON.

Required fields:

- `target_domain`: one of `source`, `target_h`, `target_self`
- `objective`: e.g. `bandstop_filter`, `target_spectrum_match`
- `frequency_min_thz`
- `frequency_max_thz`
- `constraints`
- `budget`
- `success_thresholds`

Do not invent numerical values when the user did not provide them. Use `null`
and add the missing field to `questions`.

