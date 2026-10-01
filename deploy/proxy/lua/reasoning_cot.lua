-- body_filter per /v1/chat/completions verso vLLM.
-- 1) vLLM manda il ragionamento nel campo "reasoning", Copilot legge "reasoning_content": lo rinomina.
-- 2) Copilot (VS Code, issue microsoft/vscode #338819) rimanda il ragionamento nei turni successivi solo se il delta ha
--    un id: aggiunge "cot_id" (uno per richiesta) accanto a ogni reasoning_content non nullo.
-- Funziona sia in streaming (eventi SSE separati da riga vuota) sia nella risposta intera.
-- Nei testi JSON le virgolette sono scritte \" : la sequenza "reasoning": esiste solo come nome di campo, mai dentro un
-- contenuto, quindi la sostituzione non tocca il codice o il testo delle risposte.
local ctx = ngx.ctx
local chunk, eof = ngx.arg[1], ngx.arg[2]
ctx.buf = (ctx.buf or "") .. (chunk or "")
ctx.cot_id = ctx.cot_id or ("cot_" .. ngx.var.request_id)

local function fix(ev)
    ev = ev:gsub('"reasoning":', '"reasoning_content":')
    ev = ev:gsub('"reasoning_content":"', '"cot_id":"' .. ctx.cot_id .. '","reasoning_content":"')
    return ev
end

local out = {}
while true do
    local i = ctx.buf:find("\n\n", 1, true)
    if not i then break end
    out[#out + 1] = fix(ctx.buf:sub(1, i + 1))
    ctx.buf = ctx.buf:sub(i + 2)
end
if eof then
    out[#out + 1] = fix(ctx.buf)
    ctx.buf = ""
end
ngx.arg[1] = table.concat(out)
