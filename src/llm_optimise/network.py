"""Explicit destination handling for model requests."""

import ipaddress
import socket
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, build_opener


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise HTTPError(req.full_url, code, "model endpoint redirects are refused", headers, fp)


def open_request(request, timeout):
    return build_opener(NoRedirect).open(request, timeout=timeout)


def validate_local_destination(url):
    host = urlparse(url).hostname
    if not host:
        raise ValueError("local endpoint has no host")
    addresses = socket.getaddrinfo(host, urlparse(url).port or 80, type=socket.SOCK_STREAM)
    if not addresses:
        raise ValueError("local endpoint did not resolve")
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0].split("%")[0])
        if not (ip.is_loopback or ip.is_private) or ip.is_unspecified or ip.is_multicast:
            raise ValueError(
                "local placement requires a loopback or private-network endpoint; label public providers cloud"
            )
