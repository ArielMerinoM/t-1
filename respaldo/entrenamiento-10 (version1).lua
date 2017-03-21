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
	for j = 1, #viajes[i] - 12 do
		dd = viajes[i][j][1]
		 t = viajes[i][j][2]
		 d = viajes[i][j][3]
		 y = viajes[i][j+10][3] --la funcion objetivo intenta obtener el porcentaje de avance del bus en el minuto t+10
		--normalizacion
		table.insert(X, torch.DoubleTensor({T/86400, t/86400, dd/D, d/D})) --t0, t, dd, d
		table.insert(Y, torch.DoubleTensor({100*y/D})) --d(t+1)
	end
	table.insert(datos, viajes[i][#viajes[i]])
	table.insert(viajesX, X)
	table.insert(viajesY, Y)
end


viajes = nil
collectgarbage()

print('Separando datos de validacion')

--separacion de los datos para el conjunto de validacion

n_validacion = math.floor(0.2*#viajesX)

validacionX = {}
validacionY = {}
validacionD = {}

for i = 1, n_validacion do
	indice = math.random(#viajesX)
	X = table.remove(viajesX, indice)
	Y = table.remove(viajesY, indice)
	D = table.remove(datos, indice)

	table.insert(validacionX, X)
	table.insert(validacionY, Y)
	table.insert(validacionD, D)
end

datos = nil
collectgarbage()

--registro de los viajes usados para validar

escritura_validacion = io.open('conjunto_validacion-10.txt','w')
for i = 1, #validacionD do
	escritura_validacion:write(validacionD[i][1]..'\t'..validacionD[i][2]..'\t'..validacionD[i][3]..'\t'..validacionD[i][4]..'\t'..validacionD[i][5]..'\n')
end
escritura_validacion:close()

validacionD = nil
collectgarbage()

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

red:add(nn.Sequencer(nn.LSTM(4,100,1)))
red:add(nn.Sequencer(nn.LSTM(100,100,1)))
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

registro = io.open('./registro/entrenamiento-10.txt','a')
registro:write('0\t0\t'..errFold..'\n')
registro:close()

torch.save('./redes/prediccion-10.rn', red)



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
			viajeX = foldsX[j][k] --fold j viaje k
			viajeY = foldsY[j][k]

			x = {}
			y = {}
			
			for i = 0, #viajeX do
				table.insert(x, viajeX[i])
				table.insert(y, viajeY[i])
				gradientUpgrade(red, x, y, criterio, lr)
				red:forget()
			end
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
		registro = io.open('./registro/entrenamiento-10.txt','a')
		registro:write(i..'\t'..j..'\t'..errFold..'\n')
		registro:close()

		torch.save('./redes/prediccion-10.rn', red)

	end
	errEpoca = errFold
	shuffle(foldsX,foldsY)
end


