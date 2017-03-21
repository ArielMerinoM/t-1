require 'rnn'
require 'math'
require 'funciones'

math.randomseed(os.time())

epocas = 1000
lr = 0.000001

print('Abriendo archivos')

CARPETAS = {'../Datos procesados/semana/','../Datos procesados/viernes/'}
MESES = {'03-MARZO','04-ABRIL','05-MAYO','06-JUNIO','07-JULIO','08-AGOSTO','09-SEPTIEMBRE','10-OCTUBRE','11-NOVIEMBRE','12-DICIEMBRE'}
FERIADOS = {{1,1},{2,2},{4,3},{4,21},{5,1},{9,7},{10,12},{11,2},{12,25}}

viajes = {}
for c = 1, #CARPETAS do
	for m = 1, #MESES do
		viaje = {}
		
		print(CARPETAS[c]..MESES[m]..'-I.txt')
		archivo = io.open(CARPETAS[c]..MESES[m]..'-I.txt','r')
		
		linea = archivo:read()

		while linea ~= nil do
			linea = split(linea, '\t')
			if linea[1] ~= 'F' then
				dd = tonumber(linea[1])
				 t = tonumber(linea[4])
				 d = tonumber(linea[3])
				table.insert(viaje, {dd, t, d})
			else
				if #viaje <= 91 then --si la duracion del viaje es de hasta 90 minutos
					if not feriado(FERIADOS,tonumber(linea[3]),tonumber(linea[2])) then
						T = tonumber(linea[5])*3600 + tonumber(linea[6])*60 + tonumber(linea[7])
						D = tonumber(linea[9])
						F = {tonumber(linea[2]),tonumber(linea[3]),tonumber(linea[5]),tonumber(linea[6]),tonumber(linea[7])} --dia,mes,hora,minuto,segundo
						table.insert(viaje, {T,D})
						table.insert(viaje,F)
						table.insert(viajes, viaje)
					end
				end
				viaje = {}
			end
			linea = archivo:read()
		end
		archivo:close()
	end
end

print('Procesando datos')

viajesX = {}
viajesY = {}
datos = {}

for i = 1,#viajes do
	X = {}
	Y = {}
	T = viajes[i][#viajes[i]-1][1]
	D = viajes[i][#viajes[i]-1][2]
	for j = 1, #viajes[i] - 3 do
		dd = viajes[i][j][1]
		 t = viajes[i][j][2]
		 d = viajes[i][j][3]
		 y = viajes[i][j+1][1] --la funcion objetivo intenta obtener el delta de avance del bus en el minuto t+1
		--normalizacion
		table.insert(X, torch.DoubleTensor({T/86400, t/86400, dd/D, d/D})) --t0, t, dd, d
		table.insert(Y, torch.DoubleTensor({100*y/D})) --dd(t+1) 1 - 100
	end
	table.insert(datos, viajes[i][#viajes[i]])
	table.insert(viajesX, X)
	table.insert(viajesY, Y)
end


viajes = nil
collectgarbage()

print('Separando datos de validacion')

--seleccion del conjunto de validacion
archivo_validacion = io.open('conjunto_validacion.txt','r')

linea = archivo_validacion:read()
validacionX = {}
validacionY = {}

while linea ~= nil do
	linea = split(linea, '\t')
	linea[1] = tonumber(linea[1])
	linea[2] = tonumber(linea[2])
	linea[3] = tonumber(linea[3])
	linea[4] = tonumber(linea[4])
	linea[5] = tonumber(linea[5])
	for i = 1, #datos do
		if datos[i][1] == linea[1] and datos[i][2] == linea[2] and datos[i][3] == linea[3] and datos[i][4] == linea[4] and datos[i][5] == linea[5] then
			table.insert(validacionX, viajesX[i])
			table.insert(validacionY, viajesY[i])
			table.remove(viajesX,i)
			table.remove(viajesY,i)
			table.remove(datos,i)
			break
		end
	end
	linea = archivo_validacion:read()
end

--asignacion de los folds

print('Generando folds')

foldsX = {{},{},{},{},{},{},{},{},{},{}}
foldsY = {{},{},{},{},{},{},{},{},{},{}}

while #viajesX > 0 do
	for i = 1, 10 do
		if #viajesX == 0 then
			break
		else
			indice = math.random(#viajesX)
			table.insert(foldsX[i], table.remove(viajesX, indice))
			table.insert(foldsY[i], table.remove(viajesY, indice))
			table.remove(viajesX, indice)
			table.remove(viajesY, indice)
		end
	end
end

print('Generando red')
--criterio de pruebas
tester = nn.MSECriterion()

--criterio de aprendizaje
criterio = nn.SequencerCriterion(nn.MSECriterion())

--estructura de la red neuronal
red = nn.Sequential()

red:add(nn.Sequencer(nn.LSTM(4,100,90)))
red:add(nn.Sequencer(nn.LSTM(100,100,90)))
red:add(nn.Sequencer(nn.Linear(100,100)))
red:add(nn.Sequencer(nn.Linear(100,100)))
red:add(nn.Sequencer(nn.Linear(100,1)))

--validacion inicial

print('Realizando validacion inicial...')
acumulador = 0
puntos = 0
for v = 1, #validacionX do
	viajeX = validacionX[v]
	viajeY = validacionY[v]
	resultados = red:forward(viajeX)
	red:forget()
	
	for r = 1, #resultados do
		acumulador = acumulador + tester:forward(resultados[r], viajeY[r])
		puntos = puntos + 1
	end
	if v % 10 == 0 then 
		os.execute('clear')
		print('Realizando validacion inicial...', math.floor(100*v/#validacionX)..'%')
	end		

end

errFold = acumulador / puntos
errEpoca = acumulador / puntos

registro = io.open('./registro/entrenamiento-1.txt','a')
registro:write('0\t0\t'..errFold..'\n')
registro:close()

torch.save('./redes/prediccion-1.rn', red)



for i = 1, epocas do
	for j = 1, 10 do
		for k = 1, #foldsX[j] do
			if k % 10 == 0 then
				os.execute('clear')
				print('Realizando entrenamiento')
				print('Epoca:',i-1)
				print('Fold:',j-1, math.floor(100*k/#foldsX[j])..'%')
				print('Error validacion epoca [MSE]:', errEpoca)
				print('Error validacion fold [MSE]:', errFold)
			end
			viajeX = foldsX[j][k]
			viajeY = foldsY[j][k]
			
			gradientUpgrade(red, viajeX, viajeY, criterio,lr)
			red:forget()
		end
		--validacion de resultados
		--validdacionX y validacionY --solo contienen viajes

		print('Realizando validacion...')
		acumulador = 0
		puntos = 0
		for v = 1, #validacionX do
			viajeX = validacionX[v]
			viajeY = validacionY[v]
			resultados = red:forward(viajeX)
			red:forget()
			
			for r = 1, #resultados do
				acumulador = acumulador + tester:forward(resultados[r], viajeY[r])
				puntos = puntos + 1
			end				
			if v % 50 == 0 then 
				os.execute('clear')
				print('Epoca:',i-1)
				print('Fold:',j-1)
				print('Error validacion epoca [MSE]:', errEpoca)
				print('Error validacion fold [MSE]:', errFold)
				print('Realizando validacion...', math.floor(100*v/#validacionX)..'%')
			end
		end
		errFold = acumulador / puntos

		print('Guardando, no detener')
		registro = io.open('./registro/entrenamiento-1.txt','a')
		registro:write(i..'\t'..j..'\t'..errFold..'\n')
		registro:close()

		torch.save('./redes/prediccion-1.rn', red)

	end
	errEpoca = errFold
	shuffle(foldsX,foldsY)
end


