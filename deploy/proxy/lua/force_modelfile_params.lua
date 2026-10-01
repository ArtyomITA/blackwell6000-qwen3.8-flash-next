-- Forza i parametri di sampling raccomandati per Qwen3.8 in modalita thinking (Qwen / Unsloth):
-- temperature 1.0, top_p 0.95, top_k 20, presence_penalty 0, frequency_penalty 0.
-- 19/09/2026: rimosse le regole di "pulizia virgole" che agivano su tutto il body e alteravano
-- il codice contenuto nei messaggi (es. "[a, b, ]" -> "[a, b]", "m[:, ]" -> "m[:]").
-- Le rimozioni qui sotto tolgono gia la virgola adiacente, quindi la pulizia non serve.
-- Rete di sicurezza: se il body riscritto non e JSON valido, si inoltra l'originale intatto.
local cjson = require "cjson.safe"

ngx.req.read_body()
local original = ngx.req.get_body_data()
if not original or original == "" then
    return
end
local body = original

body = body:gsub('"temperature"%s*:%s*[%d%.]+', '"temperature":1.0')

body = body:gsub('"top_p"%s*:%s*[%d%.]+', '"top_p":0.95')
if not body:find('"top_p"', 1, true) then
    body = body:gsub('("temperature"%s*:%s*[%d%.]+)', '%1,"top_p":0.95', 1)
end

body = body:gsub('"top_k"%s*:%s*%d+', '"top_k":20')

body = body:gsub(',%s*"presence_penalty"%s*:%s*[%d%.%-]+', '')
body = body:gsub('"presence_penalty"%s*:%s*[%d%.%-]+%s*,', '')
body = body:gsub(',%s*"frequency_penalty"%s*:%s*[%d%.%-]+', '')
body = body:gsub('"frequency_penalty"%s*:%s*[%d%.%-]+%s*,', '')
body = body:gsub('"model"%s*:%s*"[^"]*"', '%0,"presence_penalty":0,"frequency_penalty":0', 1)

if body ~= original then
    local ok = cjson.decode(body)
    if ok == nil then
        ngx.log(ngx.WARN, "force_modelfile_params: body riscritto non valido, inoltro l'originale")
        return
    end
    ngx.req.set_body_data(body)
    ngx.req.set_header("Content-Length", #body)
end
