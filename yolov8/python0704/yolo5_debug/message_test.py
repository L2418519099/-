# -*- coding: utf-8 -*-
# This file is modified to include a function for setting the 'code' dynamically.

import os
from alibabacloud_dysmsapi20170525.client import Client as Dysmsapi20170525Client
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_dysmsapi20170525 import models as dysmsapi_20170525_models
from alibabacloud_tea_util import models as util_models
from alibabacloud_tea_util.client import Client as UtilClient


class SmsService:
    def __init__(self):
        self.client = self._create_client()

    def _create_client(self) -> Dysmsapi20170525Client:
        """
        Initializes the SMS API client using AK&SK.
        """
        config = open_api_models.Config(
            access_key_id='LTAI5tH4pyRZ31YdaDNfJ7W7',
            access_key_secret='05vD9eFl3wkMP5WZA6YjvL79KrhOHP',
            endpoint='dysmsapi.aliyuncs.com'
        )
        return Dysmsapi20170525Client(config)

    def set_code_and_send_sms(self, phone_number: str, sign_name: str, template_code: str, code_value: str) -> dict:
        """
        Sets the 'code' value and sends an SMS message.

        :param phone_number: Recipient's phone number.
        :param sign_name: Signature name registered in Alibaba Cloud.
        :param template_code: Template code obtained from Alibaba Cloud.
        :param code_value: The string value to be used as the 'code'.
        :return: Response from the SMS API call.
        """
        template_params = {"code": code_value}  # Automatically sets the 'code' with the provided value
        request = dysmsapi_20170525_models.SendSmsRequest(
            phone_numbers=phone_number,
            sign_name=sign_name,
            template_code=template_code,
            template_param=str(template_params)
        )
        try:
            response = self.client.send_sms_with_options(request, util_models.RuntimeOptions())
            return response.body
        except Exception as error:
            print(f"Error sending SMS: {error.message}")
            print(f"Recommendation: {error.data.get('Recommend') if error.data else 'N/A'}")
            raise
    def send_warning(self):
        phone = '+8615072240918'
        sign = '阿里云短信测试'
        template_code = 'SMS_154950909'
        code_to_send = "1234"  # Replace with any string you want to send as the code

        try:
            response = self.set_code_and_send_sms(phone, sign, template_code, code_to_send)
            print(response)
        except Exception as e:
            print(f"Failed to send SMS: {e}")

# Usage example
if __name__ == '__main__':
    sms_service = SmsService()
    phone = '+8615072240918'
    sign = '阿里云短信测试'
    template_code = 'SMS_154950909'
    code_to_send = "1234"  # Replace with any string you want to send as the code

    try:
        response = sms_service.set_code_and_send_sms(phone, sign, template_code, code_to_send)
        print(response)
    except Exception as e:
        print(f"Failed to send SMS: {e}")
