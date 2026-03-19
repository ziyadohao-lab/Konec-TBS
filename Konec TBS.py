import streamlit as st

st.markdown("""
<style>
            
.stApp {
    background-image: url("https://raw.githubusercontent.com/ziyadohao-lab/Konec-TBS/main/Background.png")
    background-size: cover;
    background-position: center; 
    background-repeat: no-repeat; 
}
            
div.stButton > button {
    width: 100% !important;
    display: flex;
    justify-content: flex-start;
    text-align: left;
}

</style>
""", unsafe_allow_html=True)

st.image("logo.png", width=200)
st.set_page_config(page_title="Trouble shooting", page_icon="🔧", layout="wide")

st.title("Trouble Shooting Tool")

if "step" not in st.session_state:
    st.session_state.step = 0
if "code" not in st.session_state:
    st.session_state.code = None

#Result
if st.session_state.code is not None:

    code = st.session_state.code

    st.success(f"Diagnosis Report:")

    if code == 1:
        st.write("Contact Konec.")

    elif code == 2:
        st.write("Turn on circuit break.")
        st.write("If the gateway remains offline, check the Access Point (AP) status.")

    elif code == 3:
        st.write("Contact Superloop firstly. If it is not AP issue, then contact Konec.")

    elif code == 4:
        st.write("Contact Superloop firstly. If it is not AP issue, then contact Konec.")

    elif code == 5:
        st.write("Contact a locksmith to replace the lock.")

    elif code == 6:
        st.write("Performed a restart by removing all the batteries, reinstall the batteries after 1 min.")
        st.write("Confirm that the battery has no leakage.")
        st.write("If the issue persists, please reconfigure the door lock.(configuration step - video)")
        st.write("If the problem continues after reconfiguration, please contact a locksmith to replace the lock.")

    elif code == 7:
        st.write("Contact a locksmith to replace the lock.")

    elif code == 8:
        st.write("Contact a locksmith to replace the lock.")

    elif code == 9:
        st.write("Ensured batteries are properly installed. If the issue persists, replaced them with new batteries.")
        st.write("Confirm that the battery has no leakage.")
        st.write("If the issue persists, replaced them with new batteries.")
        st.write("If the issue persists, please reconfigure the door lock.(configuration step - video)")
        st.write("If the problem continues after reconfiguration, please contact a locksmith to replace the lock")

    elif code == 10:
        st.write("Contact Konec.")

    elif code == 11:
        st.write("Turn on circuit break.")
        st.write("If the device remains offline, please reconfigure the device.")
        st.write("If the problem continues after reconfiguration, please contact an electrician to replace the device.")


    elif code == 12:
        st.write("The issue could be either a wiring problem or a faulty device that needs replacement.")
        st.write("Contact an electrician. ")

    elif code == 13:
        st.write("Restart the device by turning the circuit breaker off and on.")
        st.write("If the device remains offline, please reconfigure the device.")
        st.write("If the problem continues after reconfiguration, please contact an electrician to replace the device.")

    elif code == 14:
        st.write("Restart the device by turning the circuit breaker off and on.")
        st.write("If the device remains offline, please reconfigure the device.")
        st.write("If the problem continues after reconfiguration, please contact an electrician to replace the device.")

    elif code == 15:
        st.write("Remove the panel cover and check that the screws are in good condition and the base is installed evenly. If everything is normal, wait 30 seconds before reattaching the cover to restart the device.")
        st.write("The following images show possible abnormal installation conditions.")
        st.write("If the device remains offline, please reconfigure the device.")
        st.write("If the problem continues after reconfiguration, please contact an electrician to replace the device.")

    elif code == 16:
        st.write("Contact an electrician to check the wiring or external device.")

    elif code == 17:
        st.write("Remove the panel cover, check whether the screw condition is normal and whether the base is installed evenly. ")
        st.write("The following images show possible abnormal installation conditions.")
        st.write("If everything is normal, then contact an electrician to replace the device.")

    elif code == 18:
        st.write("Remove the item, the device come back online after a while.")

    elif code == 19:
        st.write("Remove the panel cover and check that the screws are in good condition and the base is installed evenly. If everything is normal, wait 30 seconds before reattaching the cover to restart the device.")
        st.write("The following images show possible abnormal installation conditions.")
        st.write("If the device remains offline, please reconfigure the device.")
        st.write("If the problem continues after reconfiguration, please contact an electrician to replace the device.")

    elif code == 20:
        st.write("Remove the panel cover, check whether the screw condition is normal and whether the base is installed evenly. ")
        st.write("The following images show possible abnormal installation conditions.")
        st.write("If everything is normal, then contact an electrician to replace the device.")

    elif code == 21:
        st.write("Remove the panel cover, check whether the screw condition is normal and whether the base is installed evenly. ")
        st.write("The following images show possible abnormal installation conditions.")
        st.write("If everything is normal, then contact an electrician to replace the device.")

    elif code == 22:
        st.write("Turn on circuit break.")

    elif code == 23:
        st.write("Remove the panel cover and check that the screws are in good condition and the base is installed evenly. If everything is normal, wait 30 seconds before reattaching the cover to restart the device.")
        st.write("The following images show possible abnormal installation conditions.")
        st.write("If the device remains offline, please reconfigure the device.")
        st.write("If the problem continues after reconfiguration, please contact an electrician to replace the device.")

    elif code == 24:
        st.write("Remove the panel cover, check whether the screw condition is normal and whether the base is installed evenly. ")
        st.write("The following images show possible abnormal installation conditions.")
        st.write("If everything is normal, then contact an electrician to replace the device.")

    elif code == 25:
        st.write("Remove the panel cover, check whether the screw condition is normal and whether the base is installed evenly. ")
        st.write("The following images show possible abnormal installation conditions.")
        st.write("If everything is normal, then contact an electrician to replace the device.")

    elif code == 26:
        st.write("Send the lock authorization for the resident.(Guidebook, p. 39)")
        st.write("If the issue persists, please contact Konec.")

    elif code == 27:
        st.write("Send the lock authorization for the resident.(Guidebook, p. 39)")
        st.write("If the issue persists, please contact Konec.")

    elif code == 28:
        st.write("Contact a locksmith to replace the lock.")

    elif code == 29:
        st.write("Contact Konec to resend the key via management portal.")

    elif code == 30:
        st.write("Send the lock authorization for the resident.(Guidebook, p. 39)")
        st.write("If the issue persists, please contact Konec.")

    elif code == 31:
        st.write("Performed a restart by removing all the batteries, reinstall the batteries after 1 min.")
        st.write("Confirm that the battery has no leakage.")
        st.write("If the issue persists, please reconfigure the door lock.")
        st.write("If the problem continues after reconfiguration, please contact a locksmith to replace the lock.")

    elif code == 32:
        st.write("Performed a restart by removing all the batteries, reinstall the batteries after 1 min.")
        st.write("Confirm that the battery has no leakage.")
        st.write("If the issue persists, please reconfigure the door lock.")
        st.write("If the problem continues after reconfiguration, please contact a locksmith to replace the lock.")

    elif code == 33:
        st.write("Contact a locksmith to replace the lock.")

    elif code == 34:
        st.write("Performed a restart by removing all the batteries, reinstall the batteries after 1 min.")
        st.write("Confirm that the battery has no leakage.")
        st.write("If the issue persists, please reconfigure the door lock.")
        st.write("If the problem continues after reconfiguration, please contact a locksmith to replace the lock.")

    elif code == 35:
        st.write("Contact a locksmith to replace the lock.")

    elif code == 36:
        st.write("Replaced them with new batteries.")
        st.write("Please contact Konec. Konec will monitor the situation from the platform and identify the issues.")

    elif code == 37:
        st.write("Reauthorize device for the resident.")
        st.write("(Similar to the steps for reassigning lock to an existing resident on Guidebook p. 39)")
        st.write("If the issue persists, please check whether the resident is able to control A/C via control panel.")

    elif code == 38:
        st.write("Contact an electrician to check the A/C issue.")

    elif code == 39:
        st.write("Contact Konec.")

    elif code == 40:
        st.write("Please follow the troubleshooting steps for devices that fail to respond to APP control to ensure all IoT devices can be controlled via the APP.")

    elif code == 41:
        st.write("Contact Konec to set up automation / re-enable automation.")

    elif code == 42:
        st.write("Restart the device by turning the circuit breaker off and on.")
        st.write("If the issue persists, please contact an electrician to replace the device")

    elif code == 43:
        st.write("Restart the device by turning the circuit breaker off and on.")
        st.write("If the issue persists, please contact an electrician to replace the device")

    elif code == 44:
        st.write("Set device parameter correctly.(Guidebook, p. 37)")

    elif code == 45:
        st.write("Remove the panel cover and check that the screws are in good condition and the base is installed evenly. If everything is normal, wait 30 seconds before reattaching the cover to restart the device.")
        st.write("The following images show possible abnormal installation conditions.")
        st.write("If the device remains offline, please reconfigure the device.")
        st.write("If the problem continues after reconfiguration, please contact an electrician to replace the device.")

    elif code == 46:
        st.write("Set device parameter correctly.(Guidebook, p. 37)")
        st.write("Remove the panel cover and check that the screws are in good condition and the base is installed evenly. If everything is normal, wait 30 seconds before reattaching the cover to restart the device.")
        st.write("If the device remains offline, please reconfigure the device.")
        st.write("If the problem continues after reconfiguration, please contact an electrician to replace the device.")

    elif code == 47:
        st.write("Remove the panel cover, check whether the screw condition is normal and whether the base is installed evenly. ")
        st.write("The following images show possible abnormal installation conditions.")
        st.write("If everything is normal, then contact an electrician to replace the device.")

    elif code == 48:
        st.write("Remove the panel cover, check whether the screw condition is normal and whether the base is installed evenly. ")
        st.write("The following images show possible abnormal installation conditions.")
        st.write("If everything is normal, then contact an electrician to replace the device.")

    elif code == 49:
        st.write("Remove the panel cover, check whether the screw condition is normal and whether the base is installed evenly. ")
        st.write("The following images show possible abnormal installation conditions.")
        st.write("Contact an electrician to replace the device.")

    elif code == 50:
        st.write("Remove the panel cover, check whether the screw condition is normal and whether the base is installed evenly. ")
        st.write("The following images show possible abnormal installation conditions.")
        st.write("Contact an electrician to replace the device.")

    elif code == 51:
        st.write("Remove the panel cover and check that the screws are in good condition and the base is installed evenly. If everything is normal, wait 30 seconds before reattaching the cover to restart the device.")
        st.write("The following images show possible abnormal installation conditions.")
        st.write("If the device remains offline, please reconfigure the device.")
        st.write("If the problem continues after reconfiguration, please contact an electrician to replace the device.")

    elif code == 52:
        st.write("Reauthorize device for the resident.")
        st.write("(Similar to the steps for reassigning lock to an existing resident on Guidebook p. 39)")
        st.write("If the issue persists, please contact Konec.")

    elif code == 53:
        st.write("Contact Konec.")

    elif code == 54:
        st.write("Contact Konec to check and reset Multi-control.")

    elif code == 55:
        st.write("Please follow the troubleshooting steps for control issue to ensure both devices are back online.")

    elif code == 56:
        st.write("Re-plug both the cable and the gateway power adapter.")
        st.write("If the issue persists, please contact Konec to check and reset multi-control.")

    elif code == 57:
        st.write("Please follow the troubleshooting steps for control issue to ensure both devices are back online.")

    elif code == 58:
        st.write("Contact Konec to check and reset automation")

    elif code == 59:
        st.write("Re-plug both the cable and the gateway power adapter.")
        st.write("If the issue persists, please contact Konec to check and reset automation")

    elif code == 60:
        st.write("Contact an electrician to check the wiring or replace the device.")

    elif code == 61:
        st.write("The issue appears to be with the external device, not Konec product.")

    elif code == 62:
        st.write("Please contact Konec and provide the resident's account, app logs, and phone model.")

    elif code == 63:
        st.write("Add public area device authorizations for the room.")
        st.write("Add public area device authorizations for the resident.")
        st.write("If the issue persists, please check whether the resident has been granted access in Innerrange.")
        st.write("If there is an authorization issue with Innerrange, please contact Konec and provide the resident's account, app logs, and phone model.")

    elif code == 64:
        st.write("Add public area device authorizations for the resident.")
        st.write("If the issue persists, please check whether the resident has been granted access in Innerrange.")
        st.write("If there is an authorization issue with Innerrange, please contact Konec and provide the resident's account, app logs, and phone model.")

    elif code == 65:
        st.write("Contact PMS.")

    elif code == 66:
        st.write("Contact Konec.")

    elif code == 67:
        st.write("Contact Konec.")

    elif code == 68:
        st.write("Contact PMS.")
    
    else:
        st.write("Pending, waiting for update")

    if st.button("Restart"):
        st.session_state.step = 1
        st.session_state.code = None
        st.session_state.Iot_num = 0
        st.session_state.num232_num = 0
        st.session_state.num274_num = 0
        st.session_state.num275_num = 0
        st.session_state.num277_num = 0
        st.session_state.product_num = 0
        st.rerun()

    st.stop()


def next_step(step):
    st.session_state.step = step
    st.rerun()

# Disclaimer
if st.session_state.step == 0:

    st.subheader("Disclaimers")
    
    st.write("")
    st.write("")

    st.write("1. Please follow all instructions carefully and ensure compliance with AS/NZS 3000 and local regulations. Improper installation or modification may void the warranty and result in safety risks. ")
    st.write("2. The information provided in this manual is for general guidance and reference purposes only. Konec Solutions Pty Ltd makes no representations or warranties, express or implied, regarding the completeness, accuracy, or suitability of the content. Product specifications, features, and visuals are subject to change without notice. ")
    st.write("3. To the extent allowed by law, Konec Solutions is not liable for any damage, injury, or loss resulting from the use of this guide or the installation process. ")
    st.write("4. This disclaimer does not affect your rights under the Australian Consumer Law or the New Zealand Consumer Guarantees Act 1993. ")
    st.write("5. This product is designed for professional installation.")
    st.write("6. Disconnect power before starting installation or maintenance.")
    st.write("7. Do not exceed the product's rated electrical capacity.")
    st.write("8. Avoid using power tools unless explicitly specified.")
    st.write("9. Use appropriate personal protective equipment (e.g., gloves, safety glasses).")
    st.write("10. Do not install in areas exposed to water or high humidity unless rated for such conditions.")
    st.write("11. All content in this App including text, images, and graphics—is the property of Konec Solutions. No part may be reproduced, translated, or modified without prior written permission. ")

    st.write("")
    st.write("")

    if st.button("I Agree", key = "green_btn"):
        next_step(1)


# Select device
if st.session_state.step == 1:

    st.subheader("Select the problematic device")

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    if col1.button("Switch"):
        st.session_state.product_num = 1
        next_step(2)

    if col2.button("Lock"):
        st.session_state.product_num = 2
        next_step(2)

    if col3.button("HPS"):
        st.session_state.product_num = 3
        next_step(2)

    if col4.button("Socket"):
        st.session_state.product_num = 1
        next_step(2)

    if col5.button("A/C Gateway"):
        st.session_state.product_num = 5
        next_step(2)

    if col6.button("HID Access"):
        next_step(350)


# Is the device online or offline?
if st.session_state.step == 2:

    st.subheader("Is the device online or offline?")

    col1, col2= st.columns(2)

    if col1.button("Online", key="green_btn"):
        next_step(200)

    if col2.button("Offline", key="red_btn"):
        next_step(3)

    st.write("")
        
    st.write("")

    if st.button("Back", key="black_btn"):
        next_step(1)


# Are all other devices in the room offline / online?
if st.session_state.step == 3:

    st.subheader("Are all other devices in the room offline / online?")

    col1, col2= st.columns(2)

    if col1.button("Online", key="green_btn"):
        next_step(7)

    if col2.button("Offline", key="red_btn"):
        next_step(4)

    st.write("")
        
    st.write("")

    if st.button("Back", key="black_btn"):
        next_step(2)


# Is Konec gateway online / offline ？
if st.session_state.step == 4:

    st.subheader("Is Konec gateway online / offline?")

    col1, col2= st.columns(2)

    if col1.button("Online", key="green_btn"):
        st.session_state.code = 1
        st.rerun()

    if col2.button("Offline", key="red_btn"):
        next_step(5)

    st.write("")
        
    st.write("")

    if st.button("Back", key="black_btn"):
        next_step(3)


# Is circuit breaker on / off?
if st.session_state.step == 5:

    st.subheader("Is circuit breaker on / off?")

    col1, col2= st.columns(2)

    if col1.button("ON", key="green_btn"):
        next_step(6)

    if col2.button("OFF", key="red_btn"):
        st.session_state.code = 2
        st.rerun()

    st.write("")
        
    st.write("")

    if st.button("Back", key="black_btn"):
        next_step(4)


# Are the AP indicator lights and cable connections proper?
if st.session_state.step == 6:

    st.subheader("Are the AP indicator lights and cable connections proper?")

    col1, col2= st.columns(2)

    if col1.button("YES", key="green_btn"):
        st.session_state.code = 4
        st.rerun()

    if col2.button("NO", key="red_btn"):
        st.session_state.code = 3
        st.rerun()

    st.write("")
        
    st.write("")

    st.image("AP Light.png", width=400)

    if st.button("Back", key="black_btn"):
        next_step(5)


# Product Category
if st.session_state.step == 7:

    st.subheader("Product Category")

    col1, col2, col3, col4= st.columns(4)

    if st.session_state.product_num == 2:
        next_step(9)

    if st.session_state.product_num == 5:
        st.session_state.code = 10
        st.rerun()

    if st.session_state.product_num == 3:
        next_step(51)

    if st.session_state.product_num == 1:
        next_step(71)

    st.write("")
        
    st.write("")

    if st.button("Back", key="black_btn"):
        next_step(3)


# Can the front panel be lit up？
if st.session_state.step == 9:

    st.subheader("Can the front panel be lit up?")

    col1, col2= st.columns(2)

    if col1.button("YES", key="green_btn"):
        next_step(10)

    if col2.button("NO", key="red_btn"):
        next_step(11)

    st.write("")
        
    st.write("")

    if st.button("Back", key="black_btn"):
        next_step(3)


# Is it possible to enter the room with any of these: PIN code, fob key, or physical key?
if st.session_state.step == 10:

    st.subheader("Is it possible to enter the room with any of these: PIN code, fob key, or physical key?")

    col1, col2= st.columns(2)

    if col1.button("YES", key="green_btn"):
        st.session_state.code = 6
        st.rerun()

    if col2.button("NO", key="red_btn"):
        st.session_state.code = 5
        st.rerun()

    st.write("")
        
    st.write("")

    if st.button("Back", key="black_btn"):
        next_step(9)


#After connecting the lock to a power bank, can the front panel be lit up？
if st.session_state.step == 11:

    st.subheader("After connecting the lock to a power bank, can the front panel be lit up?")

    col1, col2= st.columns(2)

    if col1.button("YES", key="green_btn"):
        next_step(12)

    if col2.button("NO", key="red_btn"):
        st.session_state.code = 7
        st.rerun()

    st.write("")
        
    st.write("")

    if st.button("Back", key="black_btn"):
        next_step(9)


# Is it possible to enter the room with any of these: PIN code, fob key, or physical key?
if st.session_state.step == 12:

    st.subheader("Is it possible to enter the room with any of these: PIN code, fob key, or physical key?")

    col1, col2= st.columns(2)

    if col1.button("YES", key="green_btn"):
        st.session_state.code = 9
        st.rerun()

    if col2.button("NO", key="red_btn"):
        st.session_state.code = 8
        st.rerun()

    st.write("")
        
    st.write("")

    if st.button("Back", key="black_btn"):
        next_step(11)


if st.session_state.step == 51:

    st.subheader("Is the indicator light status normal?")

    col1, col2, col3= st.columns(3)

    if col1.button("Always off"):
        next_step(52)

    if col2.button("Blinking"):
        st.session_state.code = 13
        st.rerun()

    if col3.button("YES"):
        st.session_state.code = 13
        st.rerun()

    st.write("")
        
    st.write("")

    if st.button("Back", key="black_btn"):
        next_step(3)


# Is circuit breaker on / off?
if st.session_state.step == 52:

    st.subheader("Is circuit breaker on / off?")

    col1, col2= st.columns(2)

    if col1.button("ON", key="green_btn"):
        st.session_state.code = 12
        st.rerun()

    if col2.button("OFF", key="red_btn"):
        st.session_state.code = 11
        st.rerun()

    st.write("")
        
    st.write("")

    if st.button("Back", key="black_btn"):
        next_step(51)


# Is the indicator light status normal?
if st.session_state.step == 71:

    st.subheader("Is the indicator light status normal?")

    col1, col2, col3= st.columns(3)

    if col1.button("Always off"):
        next_step(77)

    if col2.button("Blinking"):
        next_step(74)

    if col3.button("YES"):
        next_step(72)
    st.write("")
        
    st.write("")

    if st.button("Back", key="black_btn"):
        next_step(3)


# Can external devices be controlled?
if st.session_state.step == 72:

    st.subheader("Can external devices be controlled?")

    col1, col2 = st.columns(2)

    if col1.button("YES", key="green_btn"):
        st.session_state.code = 15
        st.rerun()

    if col2.button("NO", key="red_btn"):
        next_step(73)

    st.write("")
        
    st.write("")

    if st.button("Back", key="black_btn"):
        next_step(71)


# Is there an audible click from the relay when the switch is toggled?
if st.session_state.step == 73:

    st.subheader("Is there an audible click from the relay when the switch is toggled?")

    col1, col2 = st.columns(2)

    if col1.button("YES", key="green_btn"):
        st.session_state.code = 16
        st.rerun()

    if col2.button("NO", key="red_btn"):
        st.session_state.code = 17
        st.rerun()

    st.write("")
        
    st.write("")

    if st.button("Back", key="black_btn"):
        next_step(72)


# Is there any item long-pressing the button?
if st.session_state.step == 74:

    st.subheader("Is there any item long-pressing the button?")

    col1, col2 = st.columns(2)

    if col1.button("YES", key="green_btn"):
        st.session_state.code = 18
        st.rerun()

    if col2.button("NO", key="red_btn"):
        next_step(75)

    st.write("")
        
    st.write("")

    if st.button("Back", key="black_btn"):
        next_step(71)


# Can external devices be controlled?
if st.session_state.step == 75:

    st.subheader("Can external devices be controlled?")

    col1, col2 = st.columns(2)

    if col1.button("YES", key="green_btn"):
        st.session_state.code = 19
        st.rerun()

    if col2.button("NO", key="red_btn"):
        next_step(76)

    st.write("")
        
    st.write("")

    if st.button("Back", key="black_btn"):
        next_step(74)


# Can external devices be controlled?
if st.session_state.step == 76:

    st.subheader("Can external devices be controlled?")

    col1, col2 = st.columns(2)

    if col1.button("YES", key="green_btn"):
        st.session_state.code = 20
        st.rerun()

    if col2.button("NO", key="red_btn"):
        st.session_state.code = 21
        st.rerun()

    st.write("")
        
    st.write("")

    if st.button("Back", key="black_btn"):
        next_step(75)


# Is circuit breaker on / off?
if st.session_state.step == 77:

    st.subheader("Is circuit breaker on / off?")

    col1, col2 = st.columns(2)

    if col1.button("ON", key="green_btn"):
        next_step(78)

    if col2.button("OFF", key="red_btn"):
        st.session_state.code = 22
        st.rerun()

    st.write("")
        
    st.write("")

    if st.button("Back", key="black_btn"):
        next_step(71)


# Can external devices be controlled?
if st.session_state.step == 78:

    st.subheader("Can external devices be controlled?")

    col1, col2 = st.columns(2)

    if col1.button("YES", key="green_btn"):
        st.session_state.code = 23
        st.rerun()

    if col2.button("NO", key="red_btn"):
        next_step(79)

    st.write("")
        
    st.write("")

    if st.button("Back", key="black_btn"):
        next_step(77)


# Can external devices be controlled?
if st.session_state.step == 79:

    st.subheader("Can external devices be controlled?")

    col1, col2 = st.columns(2)

    if col1.button("YES", key="green_btn"):
        st.session_state.code = 24
        st.rerun()

    if col2.button("NO", key="red_btn"):
        st.session_state.code = 25
        st.rerun()

    st.write("")
        
    st.write("")

    if st.button("Back", key="black_btn"):
        next_step(78)


# Product Category
if st.session_state.step == 200:

    st.subheader("Product Category")

    col1, col2, col3, col4= st.columns(4)

    if st.session_state.product_num == 2:
        next_step(201)

    if st.session_state.product_num == 5:
        next_step(231)

    if st.session_state.product_num == 3:
        next_step(251)

    if st.session_state.product_num == 1:
        next_step(271)

    st.write("")
        
    st.write("")

    if st.button("Back", key="black_btn"):
        next_step(2)


# Common Issue for Lock
if st.session_state.step == 201:

    st.subheader("Common Issue")

    if st.button("The resident is unable to open the door using the app.(Lock is online)"):
        next_step(203)

    if st.button("PIN / fob key doesn't work"):
        next_step(204)

    if st.button("Lock never locked"):
        st.session_state.code = 31
        st.rerun()

    if st.button("The numbers and symbols on the panel are not lit up"):
        next_step(207)

    if st.button("The numbers and symbols on the panel are not responding"):
        next_step(208)

    if st.button("Battery draw issue - Lock.The battery level of the door lock dropped from 100% to below 20% within one month."):
        st.session_state.code = 36
        st.rerun()

    st.write("")
        
    st.write("")

    if st.button("Back", key="black_btn"):
        next_step(2)


# Does the device display in the app?
if st.session_state.step == 203:

    st.subheader("Does the device display in the app?")

    col1, col2 = st.columns(2)

    if col1.button("YES", key="green_btn"):
        st.session_state.code = 27
        st.rerun()

    if col2.button("NO", key="red_btn"):
        st.session_state.code = 26
        st.rerun()

    st.write("")
        
    st.write("")

    if st.button("Back", key="black_btn"):
        next_step(201)


# Connecti the lock to a power bank, can the front panel be lit up?
if st.session_state.step == 204:

    st.subheader("Connecti the lock to a power bank, can the front panel be lit up?")

    col1, col2 = st.columns(2)

    if col1.button("YES", key="green_btn"):
        next_step(205)

    if col2.button("NO", key="red_btn"):
        st.session_state.code = 28
        st.rerun()

    st.write("")
        
    st.write("")

    if st.button("Back", key="black_btn"):
        next_step(201)


# Is it possible to enter the room with any of these: PIN code, fob key, or physical key?
if st.session_state.step == 205:

    st.subheader("Is it possible to enter the room with any of these: PIN code, fob key, or physical key?")

    col1, col2 = st.columns(2)

    if col1.button("YES", key="green_btn"):
        next_step(206)

    if col2.button("NO", key="red_btn"):
        st.session_state.code = 28
        st.rerun()

    st.write("")
        
    st.write("")

    if st.button("Back", key="black_btn"):
        next_step(204)


# Is it an existing PIN/fob or a new one?
if st.session_state.step == 206:

    st.subheader("Is it an existing PIN/fob or a new one?")

    col1, col2 = st.columns(2)

    if col1.button("New fob"):
        st.session_state.code = 29
        st.rerun()

    if col2.button("New PIN\nExisting PIN and fob"):
        st.session_state.code = 30
        st.rerun()

    st.write("")
        
    st.write("")

    if st.button("Back", key="black_btn"):
        next_step(205)


# Is it possible to enter the room with any of these: PIN code, fob key, or physical key?
if st.session_state.step == 207:

    st.subheader("Is it possible to enter the room with any of these: PIN code, fob key, or physical key?")

    col1, col2 = st.columns(2)

    if col1.button("YES", key="green_btn"):
        st.session_state.code = 32
        st.rerun()

    if col2.button("NO", key="red_btn"):
        st.session_state.code = 33
        st.rerun()

    st.write("")
        
    st.write("")

    if st.button("Back", key="black_btn"):
        next_step(201)


# Is it possible to enter the room with any of these: PIN code, fob key, or physical key?
if st.session_state.step == 208:

    st.subheader("Is it possible to enter the room with any of these: PIN code, fob key, or physical key?")

    col1, col2 = st.columns(2)

    if col1.button("YES", key="green_btn"):
        st.session_state.code = 34
        st.rerun()

    if col2.button("NO", key="red_btn"):
        st.session_state.code = 35
        st.rerun()

    st.write("")
        
    st.write("")

    if st.button("Back", key="black_btn"):
        next_step(201)


# Common Issue for AC
if st.session_state.step == 231:

    st.subheader("Common Issue")

    if st.button("The resident is unable to control A/C via APP"):
        st.session_state.num232_num = 1
        next_step(232)


    st.write("")
        
    st.write("")

    if st.button("Back", key="black_btn"):
        next_step(2)


# Does the device display in the app?
if st.session_state.step == 232:

    st.subheader("Does the device display in the app?")

    col1, col2 = st.columns(2)

    if col1.button("YES", key="green_btn"):
        next_step(233)

    if col2.button("NO", key="red_btn"):
        st.session_state.code = 37
        st.rerun()

    st.write("")
        
    st.write("")

    if st.button("Back", key="black_btn"):
        if st.session_state.num232_num == 1:
            next_step(231)
        else:
            next_step(256)


# Is the resident able to control A/C via control panel?
if st.session_state.step == 233:

    st.subheader("Is the resident able to control A/C via control panel?")

    col1, col2 = st.columns(2)

    if col1.button("YES", key="green_btn"):
        st.session_state.code = 39
        st.rerun()

    if col2.button("NO", key="red_btn"):
        st.session_state.code = 38
        st.rerun()

    st.write("")
        
    st.write("")

    if st.button("Back", key="black_btn"):
        next_step(232)


# Common Issue for HPS
if st.session_state.step == 251:

    st.subheader("Common Issue")

    if st.button("Automation is not working: The IoT device does not turned off automatically in the absence of people."):
        st.session_state.Iot_num = 1
        next_step(252)

    if st.button("Automation is not behaving correctly: The IoT devices automatically turns off after a period of time even n the presence of people."):
        st.session_state.Iot_num = 2
        next_step(254)

    st.write("")
        
    st.write("")

    if st.button("Back", key="black_btn"):
        next_step(2)


# Are the IoT devices able to be controlled IoT via APP?
if st.session_state.step == 252:

    st.subheader("Are the IoT devices able to be controlled IoT via APP?")

    col1, col2 = st.columns(2)

    if col1.button("YES", key="green_btn"):
        next_step(253)

    if col2.button("NO", key="red_btn"):
        next_step(256)

    st.write("")
        
    st.write("")

    if st.button("Back", key="black_btn"):
        next_step(251)


# Is the automation set up correctly / abled?
if st.session_state.step == 253:

    st.subheader("Is the automation set up correctly / abled?")

    col1, col2 = st.columns(2)

    if col1.button("YES", key="green_btn"):
        next_step(255)

    if col2.button("NO", key="red_btn"):
        st.session_state.code = 41
        st.rerun()

    st.write("")
        
    st.write("")

    if st.button("Back", key="black_btn"):
        next_step(252)


# Is the parameter set correctly? Low sensitivity - 4dm
if st.session_state.step == 254:

    st.subheader("Is the parameter set correctly? Low sensitivity - 4dm")

    col1, col2 = st.columns(2)

    if col1.button("YES", key="green_btn"):
        next_step(255)

    if col2.button("NO", key="red_btn"):
        st.session_state.code = 44
        st.rerun()

    st.write("")
        
    st.write("")

    if st.button("Back", key="black_btn"):
        next_step(251)


# Check the device status
if st.session_state.step == 255:

    st.subheader("Check the device status")

    col1, col2 = st.columns(2)

    if col1.button("Always precent / abcent"):
        st.session_state.code = 42
        st.rerun()

    if col2.button("Normal"):
        st.session_state.code = 43
        st.rerun()

    st.write("")
        
    st.write("")

    if st.button("Back", key="black_btn"):
        if st.session_state.Iot_num == 1:
            next_step(253)
        else:
            next_step(254)


# Are the IoT devices able to be controlled IoT via APP?
if st.session_state.step == 256:

    st.subheader("Are the IoT devices able to be controlled IoT via APP?")

    col1, col2 = st.columns(2)

    if col1.button("A/C Gateway"):
        st.session_state.num232_num = 2
        next_step(232)

    if col2.button("Switch/Socket"):
        st.session_state.num277_num = 3
        next_step(277)

    st.write("")
        
    st.write("")

    if st.button("Back", key="black_btn"):
        next_step(252)


# Common Issue for Lock
if st.session_state.step == 271:

    st.subheader("Common Issue")

    if st.button("The indicator light status is abnormal. (Always On / Off)"):
        next_step(272)

    if st.button("Traditional control issue - Switch, Socket. e.g. Switch cannot control the light"):
        st.session_state.num275_num = 1
        next_step(275)

    if st.button("The resident is unable to control device using the app."):
        st.session_state.num277_num = 1
        next_step(277)

    if st.button("Multi-control issue. Multi-Gang Switch & Mainlight Switch"):
        next_step(280)

    if st.button("Automation control issue - Kitchen Switch & Cooktop Switch"):
        next_step(282)

    if st.button("GPO type-C changer doesn't work"):
        next_step(284)

    st.write("")
        
    st.write("")

    if st.button("Back", key="black_btn"):
        next_step(2)


# Can external devices be controlled?
if st.session_state.step == 272:

    st.subheader("Can external devices be controlled?")

    col1, col2 = st.columns(2)

    if col1.button("YES", key="green_btn"):
        next_step(273)

    if col2.button("NO", key="red_btn"):
        st.session_state.num274_num = 1
        next_step(274)

    st.write("")
        
    st.write("")

    if st.button("Back", key="black_btn"):
        next_step(271)


# Check the device status
if st.session_state.step == 273:

    st.subheader("Is the indicator mode setting on the management portal correctly?\n(Similar to the steps for Human Presence Sensor Parameter on Guidebook p. 37)")

    col1, col2 = st.columns(2)

    if col1.button("YES", key="green_btn"):
        st.session_state.code = 45
        st.rerun()

    if col2.button("NO", key="red_btn"):
        st.session_state.code = 46
        st.rerun()

    st.write("")
        
    st.write("")

    if st.button("Back", key="black_btn"):
        next_step(272)


# Is there an audible click from the relay when the switch is toggled?
if st.session_state.step == 274:

    st.subheader("Is there an audible click from the relay when the switch is toggled?")

    col1, col2 = st.columns(2)

    if col1.button("YES", key="green_btn"):
        st.session_state.code = 47
        st.rerun()

    if col2.button("NO", key="red_btn"):
        st.session_state.code = 48
        st.rerun()

    st.write("")
        
    st.write("")

    if st.button("Back", key="black_btn"):
        if st.session_state.num274_num == 1:
            next_step(272)
        else:
            next_step(275)


# Is the indicator light status normal?
if st.session_state.step == 275:

    st.subheader("Is the indicator light status normal?")

    col1, col2 = st.columns(2)

    if col1.button("YES", key="green_btn"):
        next_step(276)


    if col2.button("NO", key="red_btn"):
        st.session_state.num274_num = 2
        next_step(274)

    st.write("")
        
    st.write("")

    if st.button("Back", key="black_btn"):
        if st.session_state.num275_num == 1:
            next_step(271)
        else:
            next_step(278)
        

# Is there an audible click from the relay when the switch is toggled?
if st.session_state.step == 276:

    st.subheader("Is there an audible click from the relay when the switch is toggled?")

    col1, col2 = st.columns(2)

    if col1.button("YES", key="green_btn"):
        st.session_state.code = 49
        st.rerun()

    if col2.button("NO", key="red_btn"):
        st.session_state.code = 50
        st.rerun()

    st.write("")
        
    st.write("")

    if st.button("Back", key="black_btn"):
        next_step(275)


# Does the device display in the app?
if st.session_state.step == 277:

    st.subheader("Does the device display in the app?")

    col1, col2 = st.columns(2)

    if col1.button("YES", key="green_btn"):
        next_step(278)

    if col2.button("NO", key="red_btn"):
        next_step(279)

    st.write("")
        
    st.write("")

    if st.button("Back", key="black_btn"):
        if st.session_state.num277_num == 1:
            next_step(271)
        elif st.session_state.num277_num == 2:
            next_step(280)
        elif st.session_state.num277_num == 3:
            next_step(282)
        else:
            next_step(256)


# Can residents control external devices by operating physical buttons?
if st.session_state.step == 278:

    st.subheader("Can residents control external devices by operating physical buttons?")

    col1, col2 = st.columns(2)

    if col1.button("YES", key="green_btn"):
        st.session_state.code = 51
        st.rerun()

    if col2.button("NO", key="red_btn"):
        st.session_state.num275_num = 2
        next_step(275)

    st.write("")
        
    st.write("")

    if st.button("Back", key="black_btn"):
        next_step(271)


# Is the device authorized  to occupant details?
if st.session_state.step == 279:

    st.subheader("Is the device authorized  to occupant details?")

    col1, col2 = st.columns(2)

    if col1.button("YES", key="green_btn"):
        st.session_state.code = 53
        st.rerun()

    if col2.button("NO", key="red_btn"):
        st.session_state.code = 52
        st.rerun()

    st.write("")
        
    st.write("")

    if st.button("Back", key="black_btn"):
        next_step(277)


# Can both of these devices be controlled by the app?
if st.session_state.step == 280:

    st.subheader("Can both of these devices be controlled by the app?")

    col1, col2 = st.columns(2)

    if col1.button("YES", key="green_btn"):
        next_step(281)

    if col2.button("NO", key="red_btn"):
        st.session_state.num277_num == 2
        next_step(277)

    st.write("")
        
    st.write("")

    if st.button("Back", key="black_btn"):
        next_step(271)


# Are the automations in the room working properly?
if st.session_state.step == 281:

    st.subheader("Are the automations in the room working properly?\n(Kitchen switch on, then cooktop switch on)")

    col1, col2 = st.columns(2)

    if col1.button("YES", key="green_btn"):
        st.session_state.code = 54
        st.rerun()

    if col2.button("NO", key="red_btn"):
        st.session_state.code = 56
        st.rerun()

    st.write("")
        
    st.write("")

    if st.button("Back", key="black_btn"):
        next_step(280)


# Can both of these devices be controlled by the app?
if st.session_state.step == 282:

    st.subheader("Can both of these devices be controlled by the app?")

    col1, col2 = st.columns(2)

    if col1.button("YES", key="green_btn"):
        next_step(283)

    if col2.button("NO", key="red_btn"):
        st.session_state.num277_num = 3
        next_step(277)

    st.write("")
        
    st.write("")

    if st.button("Back", key="black_btn"):
        next_step(271)


# Are the multi-control in the room working properly?
if st.session_state.step == 283:

    st.subheader("Are the multi-control in the room working properly?\n(Multi-Gang switch on, then mainlight on)")

    col1, col2 = st.columns(2)

    if col1.button("YES", key="green_btn"):
        st.session_state.code = 58
        st.rerun()

    if col2.button("NO", key="red_btn"):
        st.session_state.code = 59
        st.rerun()

    st.write("")
        
    st.write("")

    if st.button("Back", key="black_btn"):
        next_step(282)


# Do external devices work with a traditional outlet?
if st.session_state.step == 284:

    st.subheader("Do external devices work with a traditional outlet?")

    col1, col2 = st.columns(2)

    if col1.button("YES", key="green_btn"):
        st.session_state.code = 60
        st.rerun()

    if col2.button("NO", key="red_btn"):
        st.session_state.code = 61
        st.rerun()

    st.write("")
        
    st.write("")

    if st.button("Back", key="black_btn"):
        next_step(271)



# Common Issue for HID
if st.session_state.step == 350:

    st.subheader("Common Issue")

    if st.button("Residents cannot use the HID card in the APP for access"):
        next_step(351)

    if st.button("Resident is not receiving the HID activation code"):
        next_step(355)

    if st.button("Residents cannot activate the HID card in the APP"):
        st.session_state.code = 66
        st.rerun()

    st.write("")
        
    st.write("")

    if st.button("Back", key="black_btn"):
        next_step(1)


# Check check-in status via TUYA portal
if st.session_state.step == 351:

    st.subheader("Check check-in status via TUYA portal.")

    col1, col2 = st.columns(2)

    if col1.button("YES", key="green_btn"):
        next_step(352)

    if col2.button("NO", key="red_btn"):
        st.session_state.code = 65
        st.rerun()

    st.write("")
        
    st.write("")

    if st.button("Back", key="black_btn"):
        next_step(350)


# Check from the management portal, does the resident have any authorizations for public area devices?
if st.session_state.step == 352:

    st.subheader("Check from the management portal, does the resident have any authorizations for public area devices?")

    col1, col2 = st.columns(2)

    if col1.button("YES", key="green_btn"):
        st.session_state.code = 62
        st.rerun()

    if col2.button("NO", key="red_btn"):
        next_step(354)

    st.write("")
        
    st.write("")

    if st.button("Back", key="black_btn"):
        next_step(351)


# Does the unit have any authorizations for public area devices?
if st.session_state.step == 354:

    st.subheader("Does the unit have any authorizations for public area devices?")

    col1, col2 = st.columns(2)

    if col1.button("YES", key="green_btn"):
        st.session_state.code = 64
        st.rerun()

    if col2.button("NO", key="red_btn"):
        st.session_state.code = 63
        st.rerun()

    st.write("")
        
    st.write("")

    if st.button("Back", key="black_btn"):
        next_step(352)


# Check check-in status via TUYA portal
if st.session_state.step == 355:

    st.subheader("Check check-in status via TUYA portal.")

    col1, col2 = st.columns(2)

    if col1.button("YES", key="green_btn"):
        st.session_state.code = 67
        st.rerun()

    if col2.button("NO", key="red_btn"):
        st.session_state.code = 68
        st.rerun()

    st.write("")
        
    st.write("")

    if st.button("Back", key="black_btn"):
        next_step(350)


st.html("""
    <style>

        .st-key-black_btn button {
            background-color: #000000 !important;
            color: white !important;
            border: none;
        }
        .st-key-black_btn button:hover {
            background-color: #000000 !important;
            border: none;
        }

        .st-key-green_btn button {
            background-color: #88ff91 !important;
            color: black !important;
            border: none;
        }
        .st-key-green_btn button:hover {
            background-color: #3bcf45 !important;
            border: none;
        }
        
        .st-key-red_btn button {
            background-color: #ff8888 !important;
            color: black !important;
            border: none;
        }

        .st-key-red_btn button:hover {
            background-color: #ff4444 !important;
            border: none;
        }
        
    </style>
""")