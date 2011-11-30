<!SLIDE title-slide center>
.notes first slide

# GPU Programmierung mit CUDA #
Matthias Endler

![Cube](Grafiken/Cuda-Cube.png)

<!SLIDE center>
.notes another dark side

![Videx](Grafiken/appleii-topless.jpg)

Apple II, 1977

<!SLIDE center>
.notes another dark side

![Videx](Grafiken/Videx-Videoterm-Front.jpg)

Videx Videoterm, 1981

<!SLIDE center transition=fade>
.notes another dark side

![Videx](Grafiken/Videx-Videoterm-Front-Overlay.jpg)

Videx Videoterm, 1981

<!SLIDE center>
.notes another dark side

![VisionTek GeForce 256](Grafiken/VisionTek_GeForce_256.jpg)

VisionTek Geforce 256, 1999

<!SLIDE center>
.notes another dark side

# Fixed-function pipeline #

![Architektur](Grafiken/Aufbau-Grafikkarte-Fixed-point-functions.png)

<!SLIDE center transition=fade>
.notes another dark side

# Fixed-function pipeline #

![Architektur](Grafiken/Aufbau-Grafikkarte-Fixed-point-functions2.png)


<!SLIDE center transition=fade>
.notes another dark side

# Fixed-function pipeline #

![Architektur](Grafiken/Aufbau-Grafikkarte-Fixed-point-functions3.png)

<!SLIDE center>
.notes another dark side

# Ein CUDA Programm #

![Architektur](Grafiken/cuda.png)

<!SLIDE center>

## CUDA Device Architektur

![Architektur](Grafiken/Cuda-Blocks.png)

<!SLIDE center transition=fade>

## CUDA Device Architektur

![Architektur](Grafiken/Cuda-Warp.png)

!SLIDE

## CUDA Spracherweiterungen

<table>
  <tbody>
    <tr>
      <th>Signatur</th>
      <th>Aufruf durch</th>
      <th>Ausführung auf</th>
      <th>Anmerkungen</th>
    </tr>
    <tr>
      <td class="highlight">__host__</td>
      <td>Host (CPU)</td>
      <td>Host (CPU)</td>
      <td>Standard</td>
    </tr>
    <tr>
      <td class="highlight">__global__</td>
      <td>Host (CPU)</td>
      <td>Device (GPU)</td>
      <td></td>
    </tr>
    <tr>
      <td class="highlight">__device__</td>
      <td>Device (GPU)</td>
      <td>Device (GPU)</td>
      <td></td>
    </tr>
  </tbody>
</table>

!SLIDE

## CUDA Memory Model

![Memory](Grafiken/MemoryModel.png)

!SLIDE

## CUDA Memory Model (2)
## Speicherallokierung


<table>
  <tbody>
    <tr>
      <th>Signatur</th>
      <th>Speicherort</th>
      <th>Sichtbarkeit</th>
    </tr>
    <tr>
      <td class="highlight">__device__</td>
      <td>Global Memory</td>
      <td>Programmlaufzeit</td>
    </tr>
    <tr>
      <td class="highlight">__constant__</td>
      <td>Constant Memory</td>
      <td>Programmlaufzeit</td>
      <td></td>
    </tr>
    <tr>
      <td class="highlight">__shared__</td>
      <td>Shared Memory</td>
      <td>Laufzeit des Blocks</td>
      <td></td>
    </tr>
  </tbody>
</table>
