# Step 1
sudo chmod 777 /dev/ttyACM0
# Step 2
```python
def get_at91_ports():
    from serial.tools import list_ports

    at91_ports = [port.device for port in list_ports.comports() if (port.vid, port.pid) == (0x03EB, 0x6124)]
    return at91_ports

import chipwhisperer as cw
def program_sam_firmware(hardware_type=None):
    at91_ports = get_at91_ports()
    if len(at91_ports) == 0:
        raise OSError("Could not find bootloader serial port, please see https://chipwhisperer.readthedocs.io/en/latest/firmware.html")
    if len(at91_ports) > 1:
        raise OSError("Found multiple bootloaders, please specify com port. See https://chipwhisperer.readthedocs.io/en/latest/firmware.html")

    serial_port = at91_ports[0]
    print("Found {}".format(serial_port))
    prog = cw.SAMFWLoader(None)
    prog.program(serial_port, hardware_type=hardware_type)

program_sam_firmware("cwlite")
```
## Notice:
This is the code snippet to be run to reflash the firmware of the CHipwhisphere Lite oscilloscope. Do not use jumper cables to reinitiate firmware!

For more info contact

Mabon Ninan: ninanmm@mail.uc.edu
