require 'rnn'
require 'math'

function split(inputstr, sep)
    if sep == nil then
            sep = "%s"
    end
    local t={} ; i=1
    for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
        t[i] = str
        i = i + 1
    end
    return t
end

function horaSegundo(h,m,s)
	return h*3600 + m*60 + s
end

function feriado(lista, mes, dia)
    for i = 1, #lista do
        if mes == lista[i][1] and dia == lista[i][2] then
            return true
        end
    end
    return false
end

function gradientUpgrade(model, x, y, criterion, learningRate)
        local prediction = model:forward(x)
        local err = criterion:forward(prediction, y)
        local gradOutputs = criterion:backward(prediction, y)
    model:backward(x, gradOutputs)
    model:updateParameters(learningRate)
    model:zeroGradParameters()
end

function shuffle(lista1, lista2)
    repeticiones = #lista1
    for i = 1, repeticiones do
        indice = math.random(repeticiones)
        l1 = table.remove(lista1, indice)
        l2 = table.remove(lista2, indice)
        table.insert(lista1, l1)
        table.insert(lista2, l2)
    end
end
